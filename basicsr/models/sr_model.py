import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

import math
import numpy as np
import cv2
import copy 

@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)


        if 'discriminator' in opt:
            self.net_d = build_network(opt['discriminator'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
        else:
            self.net_d = None
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
            
        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None
            
        if train_opt.get('align_opt'):
            self.cri_align = build_loss(train_opt['align_opt']).to(self.device)
            self.cri_align_loss_weight = train_opt['align_opt']["loss_weight"]
        else:
            self.cri_align = None
            
        if train_opt.get('teacher_opt'):
            self.cri_teacher = build_loss(train_opt['teacher_opt']).to(self.device)
        else:
            self.cri_teacher = None
            
        if self.cri_pix is None and self.cri_perceptual is None and self.cri_fft is None:
            raise ValueError('Pixel, perceptual and FFT losses are None.')

        if train_opt.get("ssim_opt"):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None
            
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
                
        # fft loss
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
            
        # load balancing loss for MOAReparam models
        if hasattr(self.net_g, 'get_load_balancing_loss'):
            l_balance = self.net_g.get_load_balancing_loss()
            l_total += l_balance
            loss_dict['l_balance'] = l_balance

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            
        # Reset expert statistics periodically
        if hasattr(self.net_g, 'reset_expert_stats') and current_iter % 1000 == 0:
            self.net_g.reset_expert_stats()

    def pre_process(self):
        # pad to a multiple of window_size using reflection padding
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        _, _, h, w = self.lq.size()
        pad_h = (math.ceil(h / window_size) * window_size) - h
        pad_w = (math.ceil(w / window_size) * window_size) - w
        self.mod_pad_h, self.mod_pad_w = pad_h, pad_w
        # Use reflection padding
        self.img = F.pad(self.lq, (0, pad_w, 0, pad_h), mode='reflect')
        
    # def pre_process(self):
    #     # pad to multiplication of window_size

    #     window_size = self.opt['network_g']['window_size']
    #     self.scale = self.opt.get('scale', 1)
    #     self.mod_pad_h, self.mod_pad_w, h_pad, w_pad = 0, 0, 0, 0
    #     _, _, h, w = self.lq.size()
    #     _, _, h_old, w_old = self.lq.size()
    #     if h % window_size != 0:
    #         self.mod_pad_h = window_size - h % window_size
    #         h_pad = (h_old // window_size + 1) * window_size - h_old
    #     if w % window_size != 0:
    #         self.mod_pad_w = window_size - w % window_size
    #         w_pad = (w_old // window_size + 1) * window_size - w_old
    #     # self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
    #     img_lq = torch.cat([self.lq, torch.flip(self.lq, [2])], 2)[:, :, :h_old + h_pad, :]
    #     self.img = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                            # feat = []
                            # for i in range(len(self.fea_hooks)-1):
                            #     if self.fea_hooks[i].fea != None:
                            #         if self.fea_hooks[i].fea.ndim == 3 and self.fea_hooks[i].fea.shape[2] == 180 and self.fea_hooks[i].fea.shape[0] == 1:
                            #             if len(feat) == 0:
                            #                 feat.append(self.fea_hooks[i].fea)
                            #             elif not feat[len(feat)-1].equal(self.fea_hooks[i].fea):
                            #                 feat.append(self.fea_hooks[i].fea)
                            # torch.save(feat, "SR4_feats.pth")
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # print(self.output.size())
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]


    def ensemble_inference(self, val_data, model_paths, weights=None):
        if weights is None:
            weights = [1.0] * len(model_paths)
        assert len(weights) == len(model_paths), \
            "Length of weights must match length of model_paths."

        ensemble_output = None
        total_weight = sum(weights)

        # Save original nets
        original_net_g = self.net_g
        original_net_g_ema = getattr(self, 'net_g_ema', None)  # might be None

        for idx, (model_path, w) in enumerate(zip(model_paths, weights)):
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'params_ema' in checkpoint:
                # If net_g_ema doesn't exist, create it
                if not hasattr(self, 'net_g_ema') or self.net_g_ema is None:
                    self.net_g_ema = copy.deepcopy(self.net_g)
                self.net_g_ema.load_state_dict(checkpoint['params_ema'], strict=True)
            else:
                self.net_g.load_state_dict(checkpoint['params'], strict=True)

            # Re-feed data after loading weights
            self.feed_data(val_data)

            # Self-ensemble or normal inference
            if self.opt['val'].get("self_ensemble", False):
                print('self ensemble with N models...')
                self.test_selfensemble()
            else:
                self.pre_process()
                if 'tile' in self.opt:
                    self.tile_process()
                else:
                    self.process()
                self.post_process()

            output = self.output.detach()
            if ensemble_output is None:
                ensemble_output = output * w
            else:
                ensemble_output += output * w

        ensemble_output = ensemble_output / total_weight

        # Restore original nets
        self.net_g = original_net_g
        self.net_g_ema = original_net_g_ema

        return ensemble_output


    def test_selfensemble(self):
            """
            Perform x8 self-ensemble, ensuring we still do reflection-padding so that
            the input size is valid for window-based attention.
            """
            # 1. Do your normal padding to get self.img
            def _transform(v, op):
                # Move tensor -> numpy -> do flip/transpose -> back to tensor
                v2np = v.data.cpu().numpy()
                if op == 'v':
                    tfnp = v2np[:, :, :, ::-1].copy()
                elif op == 'h':
                    tfnp = v2np[:, :, ::-1, :].copy()
                elif op == 't':
                    # Transpose height <-> width
                    tfnp = v2np.transpose((0, 1, 3, 2)).copy()
                else:
                    tfnp = v2np  # no-op if needed

                ret = torch.from_numpy(tfnp).to(self.device)
                return ret.contiguous()

            # 2. Create the list of 8 augmented inputs from self.img
            img_list = [self.lq]
            for tf in ['v', 'h', 't']:
                img_list.extend([_transform(t, tf) for t in img_list])

            # 3. Forward each augmented input
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    out_list = [self.net_g_ema(aug) for aug in img_list]
            else:
                self.net_g.eval()
                with torch.no_grad():
                    out_list = [self.net_g(aug) for aug in img_list]
                # self.net_g.train()  # Only if you want to switch back to train mode

            # 4. Invert each augmentation on the output
            for i in range(len(out_list)):
                if i > 3:
                    out_list[i] = _transform(out_list[i], 't')
                if i % 4 > 1:
                    out_list[i] = _transform(out_list[i], 'h')
                if (i % 4) % 2 == 1:
                    out_list[i] = _transform(out_list[i], 'v')

            # 5. Average the 8 outputs
            stacked = torch.stack(out_list, dim=0)
            output = torch.mean(stacked, dim=0, keepdim=False)
            self.output = output

       
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        img_format = self.opt['val'].get('img_format', 'png')


        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            # Use self-ensemble if enabled
            if self.opt['val'].get("self_ensemble", False):
                # print("############ Using self-ensemble ############")
                # Now test_selfensemble() will handle pre_process and post_process internally
                self.test_selfensemble()
            else:
                self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()


            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.{img_format}')
                imwrite(sr_img, save_img_path)
                
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
