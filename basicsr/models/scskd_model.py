# scsakd_old_model.py  — drop-in replacement

import math
from collections import OrderedDict

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

from .base_kd_model import BaseKD
from .scskd_modules import (
    SpectralAdapter,           # vit_dim -> t_dim, cnn_dim -> s_mid
    SlicedWasserstein,
    tokens_to_map,
)

# -------------------------------------------------------------------------
# Safety: ensure registered buffers own unique storage (DDP friendly)
# -------------------------------------------------------------------------
def _clone_overlapping_buffers(module, tag="module"):
    for name, buf in module.named_buffers(recurse=True):
        if buf is not None and not buf.is_leaf:
            new = buf.detach().clone()
            new.requires_grad = False
            setattr(module, name, new)




# -------------------------------------------------------------------------
# Memory-linear student-conditioned router (Top-K gather; no dense HWxN)
# -------------------------------------------------------------------------
class StudentConditionedRouter(nn.Module):
    """
    Student-conditioned attention into teacher features with Top-K sparsification.
    Memory-linear: never materializes (B, HW, N) probs/sparse tensors.
    """
    def __init__(self, s_dim, t_dim, k=8, temperature=1.0, downsample_keys: int = 1):
        super().__init__()
        self.q = nn.Conv2d(s_dim, t_dim, 1)
        self.k = int(k)
        self.tau = float(temperature)
        self.downsample = max(1, int(downsample_keys))
        self.router_gate = nn.Parameter(torch.ones(1))
        self.register_buffer("_eps", torch.tensor(1e-6), persistent=False)

    def forward(self, F_s, T_tokens_or_map):
        """
        F_s: (B, C_s, H, W)  — student feature map
        T_tokens_or_map: tokens (B, N, d) or map (B, d, Ht, Wt)
        Returns: (B, d, H, W) context in teacher token dim
        """
        B, Cs, H, W = F_s.shape

        # Queries from student -> teacher token dim
        Q = self.q(F_s).flatten(2).transpose(1, 2).contiguous()  # (B, HW, d)

        # Keys from teacher tokens/map
        if T_tokens_or_map.dim() == 4:
            Kmap = T_tokens_or_map
            if self.downsample > 1:
                Kmap = F.avg_pool2d(Kmap, self.downsample, self.downsample)
            K = Kmap.flatten(2).transpose(1, 2).contiguous()      # (B, N, d)
        else:
            K = T_tokens_or_map.contiguous()                       # (B, N, d)

        # keep autocast types aligned
        if torch.is_autocast_enabled():
            try:
                dtype = torch.get_autocast_gpu_dtype()
            except AttributeError:
                dtype = Q.dtype
            Q = Q.to(dtype=dtype)
            K = K.to(dtype=dtype)
            # L2 Normalization
            Q = F.normalize(Q, dim=-1)
            K = F.normalize(K, dim=-1)
        # print(Q.shape, K.shape)
        # logits: (B, HW, N)
        logits = (Q @ K.transpose(1, 2)) / math.sqrt(K.size(-1))

        # numerically robust Gumbel noise in fp32
        with torch.autocast(device_type='cuda', enabled=False):
            u = torch.rand_like(logits, dtype=torch.float32).clamp_(self._eps.item(), 1.0 - self._eps.item())
            g = -torch.log(-torch.log(u))
        logits = logits + g.to(logits.dtype)

        # Top-k on logits; normalize only the selected scores
        k = min(self.k, K.size(1))
        topk_vals, topk_idx = torch.topk(logits / self.tau, k=k, dim=-1)        # (B, HW, k)
        weights = torch.softmax(topk_vals, dim=-1)                               # (B, HW, k)

        # Gather selected keys and take weighted sum
       # Expand K over HW and gather along the N dimension
        K_exp  = K.unsqueeze(1).expand(-1, topk_idx.size(1), -1, -1)                 # (B, HW, N, d)
        idxexp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, K.size(-1))               # (B, HW, k, d)
        K_sel  = torch.gather(K_exp, 2, idxexp)    
        ctx = torch.einsum('bhk,bhkd->bhd', weights, K_sel)                      # (B, HW, d)

        # Back to (B, d, H, W)
        ctx = ctx.transpose(1, 2).contiguous().reshape(B, K.size(-1), H, W)
        return self.router_gate * ctx


@MODEL_REGISTRY.register()
class SCSKDModel(BaseKD):
    """SCSA-KD with multi-tap KD, AMP, and speed optimizations (native-scale KD)."""

    # ------------------------------ Init ------------------------------
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ----- KD head config parsed early (needed before setup_optimizers) -----
        kd_opt = opt.get('kd_head', {})
        student_mid_ch    = int(kd_opt.get('student_mid_channels', 64))   # aligned_c
        teacher_token_dim = int(kd_opt.get('teacher_token_dim', 180))
        num_scales        = int(kd_opt.get('num_scales', 3))
        router_topk       = int(kd_opt.get('router_topk', 8))
        router_temp       = float(kd_opt.get('router_temp', 1.0))
        router_down       = int(kd_opt.get('router_downsample_keys', 1))

        # ----- Build modules that setup_optimizers() needs to see -----
        # Core SCSAKD modules (safe to build now)
        self.router  = StudentConditionedRouter(
            s_dim=student_mid_ch, t_dim=teacher_token_dim,
            k=router_topk, temperature=router_temp, downsample_keys=router_down
        ).to(self.device)

        # Map teacher token dim -> student mid channels for loss
        # self.align_1x1 = nn.Conv2d(teacher_token_dim, student_mid_ch, 1).to(self.device)

        # Fallback projections (rarely used)
        self.fallback_proj = nn.Conv2d(3, teacher_token_dim, 1).to(self.device)
        self.student_proj  = nn.Conv2d(3, student_mid_ch, 1).to(self.device)
        self.teacher_proj  = nn.Conv2d(3, teacher_token_dim, 1).to(self.device)
        nn.init.xavier_uniform_(self.student_proj.weight); nn.init.zeros_(self.student_proj.bias)
        nn.init.xavier_uniform_(self.teacher_proj.weight); nn.init.zeros_(self.teacher_proj.bias)

        # ---- Aligners must exist before setup_optimizers() runs ----
        self.aligned_c = student_mid_ch
        self.adapter = SpectralAdapter(
            vit_dim=teacher_token_dim,   # 180
            cnn_dim=self.aligned_c,      # 64  ← change from 180 to 64
            num_scales=num_scales,
            use_gn=True
        ).to(self.device)
        self.teacher_token_dim = teacher_token_dim
        self.align_1x1 = nn.Conv2d(teacher_token_dim, self.aligned_c, 1, bias=False).to(self.device)

        if hasattr(nn, "LazyConv2d"):
            self.student_in_align = nn.LazyConv2d(self.aligned_c, kernel_size=1, bias=True).to(self.device)
        else:
            # fallback; recreated later if in_channels mismatch (see _ensure_align helper if you use it)
            self.student_in_align = nn.Conv2d(self.aligned_c, self.aligned_c, kernel_size=1, bias=True).to(self.device)
        self.teacher_in_align = nn.Conv2d(self.teacher_token_dim, self.aligned_c, kernel_size=1, bias=True).to(self.device)
        self._runtime_aligners = nn.ModuleDict()

        # Small gate for KD path
        self.kd_gate = nn.Parameter(torch.ones(1, device=self.device))

        # ----- Training toggles parsed early (used later) -----
        train_opt = opt['train']
        self.loss_schedule = train_opt.get('loss_schedule', {}) # curriculum learning schedule for loss weights
        

        self.use_amp    = bool(train_opt.get('amp', True))
        self.swd_chunk  = int(train_opt.get('swd_chunk', 16))
        self.scaler     = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.amp_dtype  = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                        else torch.float16)

        self._runtime_aligners = nn.ModuleDict()

        super().__init__(opt)


        # ---- KD aligners (after nets/opt are constructed) ----
        self.aligned_c = student_mid_ch
        self.teacher_token_dim = teacher_token_dim

        if hasattr(nn, "LazyConv2d"):
            self.student_in_align = nn.LazyConv2d(self.aligned_c, kernel_size=1, bias=True).to(self.device)
        else:
            # fallback (we will recreate if in_channels mismatch—see _ensure_align)
            self.student_in_align = nn.Conv2d(self.aligned_c, self.aligned_c, kernel_size=1, bias=True).to(self.device)
        self.teacher_in_align = nn.Conv2d(self.teacher_token_dim, self.aligned_c, kernel_size=1, bias=True).to(self.device)
        self.kd_gate = nn.Parameter(torch.ones(1, device=self.device))

        # Teacher in fp32, frozen
        self.net_t = self.net_t.float().eval()
        for p in self.net_t.parameters():
            p.requires_grad = False
        self.net_g.train()

        _clone_overlapping_buffers(self.net_g, tag="net_g")
        _clone_overlapping_buffers(self.net_t, tag="net_t")

        # Feature hooks
        self._setup_hooks()
        get_root_logger().info(
            f"[SCSAKD] Registered hook handles — T:{len(getattr(self, '_t_handles', []))}, "
            f"S:{len(getattr(self, '_s_handles', []))}"
        )
        self.cri_swd  = SlicedWasserstein(num_proj=train_opt.get('swd_num_proj', 64)).to(self.device)

        # Weights
        self.lambda_rec  = train_opt.get('lambda_rec',  1.0)
        self.lambda_kd   = train_opt.get('lambda_kd',   0.1)
        self.lambda_swd  = train_opt.get('lambda_swd',  0.2)

        # Initialize dynamic weights (will update during training)
        # Always initialize current_lambda values, regardless of loss_schedule
        self.current_lambda_rec = self.lambda_rec
        self.current_lambda_kd = self.lambda_kd  
        self.current_lambda_swd = self.lambda_swd
        
        if bool(train_opt.get('compile', False)) and hasattr(torch, 'compile'):
            self.net_g = torch.compile(self.net_g, mode="max-autotune")

        self.grad_clip_norm = train_opt.get('grad_clip_norm', 1.0)
        self.log_dict = OrderedDict()

    # ----------------------------- Optimizers -----------------------------
    def setup_optimizers(self):
        train_opt = self.opt['train']
        logger = get_root_logger()

        optim_params = []
        for n, p in self.net_g.named_parameters():
            if p.requires_grad:
                optim_params.append(p)
            else:
                logger.warning(f'Params {n} will not be optimized.')

        # Core SCSAKD modules
        optim_params += list(self.adapter.parameters())
        optim_params += list(self.router.parameters())
        optim_params += list(self.align_1x1.parameters())
        optim_params += list(self.fallback_proj.parameters())
        optim_params += list(self.student_proj.parameters())
        optim_params += list(self.teacher_proj.parameters())
        # Add aligners
        optim_params += list(self.student_in_align.parameters())
        optim_params += list(self.teacher_in_align.parameters())
        optim_params += list(self._runtime_aligners.parameters())

        optim_cfg = dict(train_opt['optim_g'])
        optim_type = optim_cfg.pop('type', 'Adam')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **optim_cfg)
        self.optimizers = [self.optimizer_g]


    # curriculum weights not used. 
    def _update_curriculum_weights(self, current_iter):
        """Update loss weights based on training schedule."""
        if not self.loss_schedule:
            return
            
        total_iter = self.opt['train']['total_iter']
        
        # Define curriculum phases (can be customized)
        if current_iter < total_iter * 0.33:      # Phase 1: 0-33%
            phase = 0
        elif current_iter < total_iter * 0.67:    # Phase 2: 33-67%  
            phase = 1
        else:                                     # Phase 3: 67-100%
            phase = 2
        
        # Update weights if schedule exists
        if 'lambda_rec' in self.loss_schedule:
            self.current_lambda_rec = self.loss_schedule['lambda_rec'][phase]
        if 'lambda_kd' in self.loss_schedule:
            self.current_lambda_kd = self.loss_schedule['lambda_kd'][phase]
        if 'lambda_swd' in self.loss_schedule:
            self.current_lambda_swd = self.loss_schedule['lambda_swd'][phase]



    def _find_by_patterns(self, model, patterns):
        """
        Match modules by exact name first; otherwise by substring.
        Returns an ordered dict {name: module} preserving pattern order.
        """
        allmods = dict(model.named_modules())
        out = {}
        for pat in patterns:
            pat = str(pat)
            if pat in allmods:                 # exact match preferred
                out[pat] = allmods[pat]
                continue
            hits = [n for n in allmods if pat in n]
            if not hits:
                continue
            name = min(hits, key=len)          # choose the shortest qualified name
            out[name] = allmods[name]
        return out

    def _name_ok_for_teacher(self, name):
        # keep heads/tails + residual_group outputs; drop internals that bloat cost
        if any(b in name for b in ('.attn', '.mlp', '.norm', '.drop_path', '.qkv', '.proj', '.blocks.')):
            return False
        if any(g in name for g in ('conv_first', 'conv_after_body', 'upsample.0', 'conv_hr', 'conv_last')):
            return True
        if re.search(r'layers\.\d+\.residual_group(\.conv)?$', name):
            return True
        # exclude the whole HR upsample block
        if re.search(r'upsample$', name):
            return False
        return False

    def _name_ok_for_student(self, name):
        # drop attention/MLP/norm internals (handles ".attention" and ".attn")
        if any(b in name for b in ('.attention', '.attn', '.mlp', '.norm')):
            return False
        if any(g in name for g in ('conv_first', 'upsample.0', 'conv_last')):
            return True
        if ('body.' in name) and (
            '.rcab' in name or name.endswith('.conv2') or re.search(r'body\.\d+\.conv$', name)
        ):
            return True
        if re.search(r'body\.\d+(\.residual_group)?$', name):
            return True
        return False

    def _select_spread(self, items, target_count):
        """Evenly spread selections across depth/order."""
        if not items:
            return {}
        items.sort(key=lambda x: (len(x[0].split('.')), x[0]))
        if len(items) <= target_count:
            return dict(items)
        step = (len(items) - 1) / max(1, (target_count - 1))
        idxs = [int(round(i * step)) for i in range(target_count)]
        return {items[i][0]: items[i][1] for i in idxs}

    def _setup_hooks(self):
        self._teacher_feats, self._student_feats = {}, {}

        dist = self.opt.get('distill', {})
        tgt = int(dist.get('target_feature_count', 6))

        t_list = dist.get('teacher_tap', []) or []
        s_list = dist.get('student_tap', []) or []
        t_manual = isinstance(t_list, (list, tuple)) and ('auto' not in [str(x).lower() for x in t_list]) and len(t_list) > 0
        s_manual = isinstance(s_list, (list, tuple)) and ('auto' not in [str(x).lower() for x in s_list]) and len(s_list) > 0

        if t_manual:
            t_hits = self._find_by_patterns(self.net_t, t_list)
        else:
            # auto teacher: filter to good layers, then spread
            t_cands = [(n, m) for n, m in self.net_t.named_modules() if n and self._name_ok_for_teacher(n)]
            t_hits = self._select_spread(t_cands, tgt)

        if s_manual:
            s_hits = self._find_by_patterns(self.net_g, s_list)
        else:
            # auto student: filter to good layers, then spread
            s_cands = [(n, m) for n, m in self.net_g.named_modules() if n and self._name_ok_for_student(n)]
            s_hits = self._select_spread(s_cands, tgt)

        # register hooks
        def _t_hook(name):
            def f(m, inp, out):
                self._teacher_feats[name] = out
            return f

        def _s_hook(name):
            def f(m, inp, out):
                self._student_feats[name] = out
            return f

        self._t_handles = [m.register_forward_hook(_t_hook(n)) for n, m in t_hits.items()]
        self._s_handles = [m.register_forward_hook(_s_hook(n)) for n, m in s_hits.items()]

        logger = get_root_logger()
        mode_T = "MANUAL" if t_manual else "AUTO"
        mode_S = "MANUAL" if s_manual else "AUTO"
        logger.info(f"[SCSAKD] Registered hook handles — Teacher({mode_T}): {len(self._t_handles)}, Student({mode_S}): {len(self._s_handles)}")
        logger.info(f"[SCSAKD] T taps: {list(t_hits.keys())}")
        logger.info(f"[SCSAKD] S taps: {list(s_hits.keys())}")
    # ----------------------------- end Hooks -----------------------------

    
        
    def _align_to(self, x, out_c, key):
        in_c = x.shape[1]
        if in_c == out_c:
            return x
        name = f"{key}_{in_c}_to_{out_c}"
        if name not in self._runtime_aligners:
            conv = nn.Conv2d(in_c, out_c, 1).to(x.device, dtype=x.dtype)
            nn.init.kaiming_uniform_(conv.weight, a=1.0)
            nn.init.zeros_(conv.bias)
            self._runtime_aligners[name] = conv
            # make sure new params are optimized
            if hasattr(self, 'optimizer_g'):
                self.optimizer_g.add_param_group({'params': conv.parameters()})
        return self._runtime_aligners[name](x)


    def _select_strategic_layers(self, modules, target_count=6):
        if len(modules) <= target_count:
            return dict(modules)
        modules.sort(key=lambda x: len(x[0].split('.')))
        selected = {}
        selected[modules[0][0]] = modules[0][1]
        selected[modules[-1][0]] = modules[-1][1]
        if len(modules) > 2:
            step = (len(modules) - 2) / (target_count - 2)
            for i in range(1, target_count - 1):
                idx = int(1 + i * step)
                selected[modules[idx][0]] = modules[idx][1]
        return selected

    # -------------------------------- IO --------------------------------
    def feed_data(self, data):
        # Apply memory format only during inference, not training
        if self.net_g.training:
            # Training: use default memory format to avoid DDP stride issues
            self.lq = data['lq'].to(self.device, non_blocking=True).float()
            self.gt = data.get('gt', None)
            if self.gt is not None:
                self.gt = self.gt.to(self.device, non_blocking=True).float()
        else:
            # Inference: can use channels_last for potential speedup
            self.lq = data['lq'].to(self.device, memory_format=torch.channels_last, non_blocking=True).float()
            self.gt = data.get('gt', None)
            if self.gt is not None:
                self.gt = self.gt.to(self.device, memory_format=torch.channels_last, non_blocking=True).float()


    def _ensure_align(self, x, out_c, base_attr, cache_prefix):
        """
        Align (B, C_in, H, W) -> (B, out_c, H, W).
        Uses the base 1x1 if it matches (or is LazyConv2d). Otherwise builds/uses a cached 1x1 for C_in->out_c.
        base_attr: 'student_in_align' or 'teacher_in_align'
        cache_prefix: short key ('salign'/'talign') for the cache.
        """
        mod = getattr(self, base_attr)
        in_c = int(x.shape[1])

        # Fast path: if base module can accept this in_channels (or is LazyConv2d), use it.
        if isinstance(mod, nn.LazyConv2d) or getattr(mod, "in_channels", in_c) == in_c:
            return mod(x)

        # Otherwise, build/reuse a dedicated 1x1 for this in_c -> out_c
        name = f"{cache_prefix}_{in_c}_to_{out_c}"
        if name not in self._runtime_aligners:
            conv = nn.Conv2d(in_c, out_c, 1).to(x.device, dtype=x.dtype)
            nn.init.kaiming_uniform_(conv.weight, a=1.0)
            nn.init.zeros_(conv.bias)
            self._runtime_aligners[name] = conv
            # Make sure new params are optimized
            if hasattr(self, "optimizer_g"):
                self.optimizer_g.add_param_group({"params": conv.parameters()})
        return self._runtime_aligners[name](x)


    def _ensure_tdim(self, x):
        """
        Make teacher map 'x' have channel = self.teacher_token_dim via a cached 1x1.
        Keeps params in optimizer. Works for any in_channels (64, 180, 192, 256, ...).
        """
        in_c = int(x.shape[1])
        if in_c == self.teacher_token_dim:
            return x
        name = f"tproj_{in_c}_to_{self.teacher_token_dim}"
        if name not in self._runtime_aligners:
            conv = nn.Conv2d(in_c, self.teacher_token_dim, 1).to(x.device, dtype=x.dtype)
            nn.init.kaiming_uniform_(conv.weight, a=1.0)
            nn.init.zeros_(conv.bias)
            self._runtime_aligners[name] = conv
            if hasattr(self, "optimizer_g"):
                self.optimizer_g.add_param_group({"params": conv.parameters()})
        return self._runtime_aligners[name](x)

    # --------------------------- Train step ------------------------------
    def optimize_parameters(self, current_iter=None):
        lq, gt = self.lq, self.gt

        self.optimizer_g.zero_grad()
        self._teacher_feats.clear()
        self._student_feats.clear()

        lq_for_teacher = lq.contiguous()
        
        with torch.no_grad():
            self.output_t = self.net_t(lq_for_teacher)
        self.output = self.net_g(lq)

        loss_dict = OrderedDict()

        # (1) Reconstruction loss
        l_rec = self.cri_pix(self.output, gt) if self.cri_pix else torch.tensor(0.0, device=self.device)
        loss_dict['l_rec'] = l_rec

        # (2) Basic KD (response)
        l_kd = self.cri_kd(self.output, self.output_t) if self.cri_kd else torch.tensor(0.0, device=self.device)
        loss_dict['l_kd'] = l_kd

        # (3) Multi-tap KD at native feature scale
        t_keys = sorted([k for k in self._teacher_feats.keys()])
        s_keys = sorted([k for k in self._student_feats.keys()])

        logger = get_root_logger()
        # Log once at start so you can see counts (rank0 only via root logger)
        if current_iter is None or current_iter < 3:
            logger.info(f"[SCSAKD] Hooks working — Teacher: {len(t_keys)}, Student: {len(s_keys)}")
            logger.info(f"[SCSAKD] T taps: {t_keys[:len(t_keys)]}")
            logger.info(f"[SCSAKD] S taps: {s_keys[:len(s_keys)]}")
        if (len(t_keys) == 0 or len(s_keys) == 0) and (current_iter is None or current_iter < 5):
            logger.info(f"Hooks captured - Teacher: {len(t_keys)}, Student: {len(s_keys)}")

        # Pairing policy
        pairing = self.opt.get('distill', {}).get('tap_pairing', 'pair_last')
        teacher_for_student = []
        if len(t_keys) == 0:
            teacher_map_fallback = tokens_to_map(self.output_t, target_hw=self.output.shape[-2:])
            if teacher_map_fallback.shape[1] != self.teacher_token_dim:
                teacher_map_fallback = self.fallback_proj(teacher_map_fallback)
            teacher_for_student = [teacher_map_fallback] * max(1, len(s_keys))
        else:
            if pairing == 'broadcast_last':
                t_ref = self._teacher_feats[t_keys[-1]]
                teacher_for_student = [t_ref] * max(1, len(s_keys))
            elif pairing == 'zip':
                L = min(len(t_keys), len(s_keys)) if len(s_keys) > 0 else len(t_keys)
                teacher_for_student = [self._teacher_feats[t_keys[i]] for i in range(L)]
                s_keys = s_keys[:L] if len(s_keys) > 0 else []
            else:  # 'pair_last'
                t_ref = self._teacher_feats[t_keys[-1]]
                teacher_for_student = [t_ref] * max(1, len(s_keys))

        if len(s_keys) == 0:
            proxy_s = F.interpolate(self.output, scale_factor=0.5, mode='bilinear', align_corners=False)
            s_feats = [('proxy', proxy_s)]
            if len(teacher_for_student) == 0:
                teacher_for_student = [tokens_to_map(self.output_t, target_hw=self.output.shape[-2:])]
        else:
            s_feats = [(k, self._student_feats[k]) for k in s_keys]

        l_swd_total = torch.zeros((), device=self.device)
        processed = 0

        for i, (s_name, s_feat) in enumerate(s_feats):
            t_feat = teacher_for_student[i if i < len(teacher_for_student) else -1]

            s_feat = s_feat.contiguous()
            t_feat = t_feat.contiguous()
            # ---- KD at native feature scale (no HR blow-up), with channel alignment ----
            if s_feat.dim() == 3:
                side = int((s_feat.shape[1]) ** 0.5)
                s_feat = tokens_to_map(s_feat, target_hw=(side, side))

            target_hw = s_feat.shape[-2:]

            if t_feat.dim() == 3:
                t_map = tokens_to_map(t_feat, target_hw=target_hw)
            else:
                t_map = t_feat if t_feat.shape[-2:] == target_hw else \
                        F.interpolate(t_feat, size=target_hw, mode='bilinear', align_corners=False)

            # ----- Channel alignment to common dim (optional if you use elsewhere) -----
            s_basic = self._ensure_align(s_feat, self.aligned_c, 'student_in_align', 'salign')
            t_basic = self._ensure_align(t_map,  self.aligned_c, 'teacher_in_align', 'talign')

            # ----- Normalize teacher tokens and align student BEFORE routing -----
            t_map   = self._ensure_tdim(t_map)                                        # (B, 180, H, W)
            s_basic = self._ensure_align(s_feat, self.aligned_c, 'student_in_align', 'salign')  # (B, 64, H, W)

            # Adapter now produces 64-ch; router returns 180-ch
            adapter_map = self.adapter(t_map, target_hw=target_hw)                    # (B, 64, H, W)
            router_ctx  = self.router(s_basic, t_map)                                 # (B, 180, H, W)

            # Project router 180 -> 64, then fuse  (LIGHT path)
            router_ctx      = self.align_1x1(router_ctx)                              # (B, 64, H, W)
            aligned_teacher = adapter_map + router_ctx           

            l_swd_i = self.cri_swd(s_basic, aligned_teacher, chunk=self.swd_chunk)

            loss_dict[f'l_swd@{s_name}'] = l_swd_i
            l_swd_total = l_swd_total + l_swd_i
            processed += 1

        l_swd_avg = (l_swd_total / processed) if processed > 0 else torch.zeros((), device=self.device)
        loss_dict['l_swd'] = l_swd_total
        
        if processed == 0:
            try:
                s_fallback = F.interpolate(self.output, scale_factor=0.5, mode='bilinear', align_corners=False)  # (B, 3, ...)
                t_fallback = tokens_to_map(self.output_t, target_hw=s_fallback.shape[-2:])                        # (B, ?, ...)

                t_fallback = self._ensure_tdim(t_fallback)                                   # (B, 180, ...)
                s_fb       = self._ensure_align(s_fallback, self.aligned_c, 'student_in_align', 'salign')  # (B, 64, ...)

                # Adapter: 64-ch; Router: 180-ch
                adapter_map = self.adapter(t_fallback, target_hw=s_fb.shape[-2:])            # (B, 64, ...)
                router_ctx  = self.router(s_fb, t_fallback)                                  # (B, 180, ...)

                # ✅ Align router to 64 BEFORE the sum (same rule as main path)
                router_ctx      = self.align_1x1(router_ctx)                                 # (B, 64, ...)
                aligned_teacher = adapter_map + router_ctx                                   # (B, 64, ...)

                l_swd_avg = self.cri_swd(s_fb, aligned_teacher, chunk=self.swd_chunk)
                loss_dict['l_swd'] = l_swd_avg
                loss_dict['l_swd_fallback'] = l_swd_avg

            except Exception:
                # Minimal basic alignment
                t_map_basic = tokens_to_map(self.output_t, target_hw=self.output.shape[-2:])
                t_map_basic = self._ensure_tdim(t_map_basic)                 # (B, 180, ...)
                s_basic     = self._ensure_align(self.output, self.aligned_c, 'student_in_align', 'salign')
                t_basic     = self.align_1x1(t_map_basic)                    # (B, 64, ...)
                l_swd_avg   = F.l1_loss(s_basic, t_basic)
                loss_dict['l_swd'] = l_swd_avg

            if l_rec == 0 and l_kd == 0:
                return

        if current_iter is not None:
            self._update_curriculum_weights(current_iter)

        # Total loss
        w = self.opt['train']
        loss_total = (
            w.get('pixel_opt', {}).get('loss_weight', 1.0) * self.current_lambda_rec * l_rec +
            w.get('kd_opt', {}).get('loss_weight', 1.0) * self.current_lambda_kd * l_kd +
            self.current_lambda_swd * l_swd_avg * self.kd_gate
        )


        loss_total.backward()
        self.optimizer_g.step()

        # Log + EMA
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # ------------------------------ Eval ------------------------------
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            out = self.net_g(self.lq.float())
            self.output = out.clamp_(0, 1).to(torch.float32).contiguous()
        self.net_g.train()

    # ----------------------------- Accessors --------------------------
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out = OrderedDict()
        out['lq'] = self.lq.detach().cpu().to(torch.float32).contiguous()
        out['result'] = self.output.detach().cpu().to(torch.float32).contiguous()
        if hasattr(self, 'gt') and self.gt is not None:
            out['gt'] = self.gt.detach().cpu().to(torch.float32).contiguous()
        return out

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def __del__(self):
        try:
            for handle in getattr(self, '_t_handles', []) + getattr(self, '_s_handles', []):
                handle.remove()
        except:
            pass
