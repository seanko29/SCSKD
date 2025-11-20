# scsa_kd_modules.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Robust tokens -> map utility
# -------------------------------

def _closest_hw(n: int):
    r = int(math.sqrt(n))
    for h in range(r, 0, -1):
        if n % h == 0:
            return h, n // h
    return r, max(1, n // max(1, r))

def tokens_to_map(tokens, target_hw=None):
    """
    Convert ViT-style tokens (B,N,C) to a 2D feature map (B,C,H,W).
    Clones input up-front to avoid any view-related overlap across ranks.
    """
    t = tokens.clone()

    # Already a map
    if t.dim() == 4:
        x = t
        if target_hw and x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x

    # Tokens -> map
    B, N, C = t.shape
    if target_hw is not None:
        Ht, Wt = target_hw
        Nexp = Ht * Wt
        if N == Nexp:
            return t.transpose(1, 2).contiguous().view(B, C, Ht, Wt)
        if N == Nexp + 1:
            t = t[:, 1:, :]
            return t.transpose(1, 2).contiguous().view(B, C, Ht, Wt)

    H0, W0 = _closest_hw(N)
    x = t.transpose(1, 2).contiguous().view(B, C, H0, W0)
    if target_hw and (H0, W0) != target_hw:
        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
    return x


# -------------------------------
# Spectral Adapter (ViT -> CNN)
# -------------------------------

class SpectralAdapter(nn.Module):
    def __init__(self, vit_dim, cnn_dim, num_scales=3, use_gn=True):
        super().__init__()
        self.num_scales = int(num_scales)

        self.proj = nn.ModuleList()
        for _ in range(self.num_scales):
            layers = [
                nn.Conv2d(vit_dim, cnn_dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(cnn_dim, cnn_dim, 3, 1, 1),
            ]
            if use_gn:
                layers.insert(1, nn.GroupNorm(1, cnn_dim))
            self.proj.append(nn.Sequential(*layers))

        self.mag_head = nn.Sequential(nn.Conv2d(cnn_dim, cnn_dim, 3, 1, 1), nn.GELU())
        self.pha_head = nn.Sequential(nn.Conv2d(cnn_dim, cnn_dim, 3, 1, 1), nn.GELU())

        # one weight per branch (scales + mag + pha)
        self.branch_logits = nn.Parameter(torch.zeros(self.num_scales + 2))
        self.post_fuse = nn.Sequential(nn.Conv2d(cnn_dim, cnn_dim, 1), nn.GELU())

    def forward(self, vit_feat, target_hw):
        aligned = []
        base = None

        for i, proj in enumerate(self.proj):
            scale = 2 ** i
            hw_i = (max(1, target_hw[0] // scale), max(1, target_hw[1] // scale))
            m = tokens_to_map(vit_feat, target_hw=hw_i)  # fresh map each scale
            a = proj(m)
            if i == 0:
                base = a
            if i > 0 and a.shape[-2:] != target_hw:
                a = F.interpolate(a, size=target_hw, mode='bilinear', align_corners=False)
            aligned.append(a)

        # FFT in fp32 for stability
        with torch.autocast(device_type='cuda', enabled=False):
            fft32 = torch.fft.fft2(base.float(), dim=(-2, -1))
            mag = torch.abs(fft32)
            # phase surrogate
            pha = torch.cos(torch.angle(fft32))

        # Out-of-place NaN/Inf guards (return new tensors)
        mag = torch.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6).to(base.dtype)
        pha = torch.nan_to_num(pha, nan=0.0, posinf=1e6, neginf=-1e6).to(base.dtype)

        mag = self.mag_head(mag.contiguous())
        pha = self.pha_head(pha.contiguous())
        aligned += [mag, pha]

        w = torch.softmax(self.branch_logits, dim=0)
        out = sum(w[i] * a for i, a in enumerate(aligned))
        return self.post_fuse(out)


# -------------------------------
# Student-Conditioned Router
# -------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # keeps the exact magnitude flexible (harmless; helps match old scaling)
        self.router_gate = nn.Parameter(torch.ones(1))
        self.register_buffer("_eps", torch.tensor(1e-6), persistent=False)

    def forward(self, F_s, T_tokens_or_map):
        B, Cs, H, W = F_s.shape

        # Q: (B, HW, t_dim)
        Q = self.q(F_s).flatten(2).transpose(1, 2)  # (B, HW, d)

        # K: (B, N, t_dim)
        if T_tokens_or_map.dim() == 4:
            Kmap = T_tokens_or_map
            if self.downsample > 1:
                Kmap = F.avg_pool2d(Kmap, self.downsample, self.downsample)
            K = Kmap.flatten(2).transpose(1, 2)     # (B, N, d)
        else:
            K = T_tokens_or_map                     # (B, N, d)

        # keep autocast types aligned
        if torch.is_autocast_enabled():
            try:
                dtype = torch.get_autocast_gpu_dtype()
            except AttributeError:
                dtype = Q.dtype
            Q = Q.to(dtype=dtype)
            K = K.to(dtype=dtype)

        # logits: (B, HW, N)
        logits = (Q @ K.transpose(1, 2)) / math.sqrt(K.size(-1))

        # numerically robust Gumbel noise in fp32
        with torch.autocast(device_type='cuda', enabled=False):
            u = torch.rand_like(logits, dtype=torch.float32).clamp_(self._eps.item(), 1.0 - self._eps.item())
            g = -torch.log(-torch.log(u))
        logits = logits + g.to(logits.dtype)

        # ---- Top-K on logits (no dense probs / no scatter) ----
        k = min(self.k, K.size(1))
        topk_vals, topk_idx = torch.topk(logits / self.tau, k=k, dim=-1)        # (B, HW, k)
        weights = torch.softmax(topk_vals, dim=-1)                               # (B, HW, k)

        # Gather only the selected keys and take weighted sum
        K_sel = K.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, K.size(-1)))   # (B, HW, k, d)
        ctx = torch.einsum('bhk,bhkd->bhd', weights, K_sel)                      # (B, HW, d)

        # back to map
        ctx = ctx.transpose(1, 2).reshape(B, K.size(-1), H, W)                   # (B, d, H, W)
        return self.router_gate * ctx


# -------------------------------
# Losses
# -------------------------------

class SlicedWasserstein(nn.Module):
    def __init__(self, num_proj=64):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=3, padding=1)
        self.num_proj = int(num_proj)

    def forward(self, A, B, eps=1e-8, chunk: int = 16):
        Ap = self.unfold(A).transpose(1, 2)  # (B, HW, 9C)
        Bp = self.unfold(B).transpose(1, 2)
        B_, HW, D = Ap.shape

        P = int(self.num_proj)
        step = max(1, int(chunk))
        loss = A.new_tensor(0.0)

        for s in range(0, P, step):
            e = min(P, s + step)
            v = torch.randn(B_, e - s, D, device=A.device, dtype=Ap.dtype)
            v = v / (v.norm(dim=-1, keepdim=True) + eps)

            Ap1 = (Ap.unsqueeze(1) @ v.transpose(1, 2)).squeeze(-1)  # (B, e-s, HW)
            Bp1 = (Bp.unsqueeze(1) @ v.transpose(1, 2)).squeeze(-1)

            Ap1_sorted, _ = torch.sort(Ap1, dim=-1)
            Bp1_sorted, _ = torch.sort(Bp1, dim=-1)
            loss = loss + torch.mean(torch.abs(Ap1_sorted - Bp1_sorted))

        return loss / math.ceil(P / step)


class FrequencyRingLoss(nn.Module):
    """
    Compares radially-binned magnitude spectra. FFT in fp32 + out-of-place NaN guards.
    Uses a small cache for integer bin masks keyed by (H, W).
    """
    def __init__(self, num_bins=8):
        super().__init__()
        self.num_bins = int(num_bins)
        self._radial_cache = {}  # (H,W) -> bins (H,W) int64 on CUDA as needed

    def _bins(self, H, W, device):
        key = (H, W, device.index if device.type == 'cuda' else -1)
        bins = self._radial_cache.get(key, None)
        if bins is None:
            with torch.no_grad():
                yy, xx = torch.meshgrid(
                    torch.arange(H, device=device, dtype=torch.float32),
                    torch.arange(W, device=device, dtype=torch.float32),
                    indexing='ij'
                )
                rr = torch.sqrt((yy - H * 0.5) ** 2 + (xx - W * 0.5) ** 2)
                rmax = torch.clamp(rr.max(), min=1.0)
                bins = torch.floor((rr / rmax) * self.num_bins).clamp(0, self.num_bins - 1).to(torch.int64)
            self._radial_cache[key] = bins
        return bins

    def forward(self, y_s, y_t, eps=1e-8):
        # FFT in fp32
        with torch.autocast(device_type='cuda', enabled=False):
            YS = torch.fft.fft2(y_s.float(), dim=(-2, -1))
            YT = torch.fft.fft2(y_t.float(), dim=(-2, -1))
            magS = torch.abs(YS)
            magT = torch.abs(YT)

        # Out-of-place NaN/Inf guards, then cast back to AMP dtype if any
        amp_dtype = y_s.dtype
        magS = torch.nan_to_num(magS, nan=0.0, posinf=1e6, neginf=-1e6).to(amp_dtype)
        magT = torch.nan_to_num(magT, nan=0.0, posinf=1e6, neginf=-1e6).to(amp_dtype)

        B, C, H, W = magS.shape
        bins = self._bins(H, W, magS.device)

        loss = magS.new_tensor(0.0)
        valid_bins = 0

        # Compute per-bin means out-of-place
        for b in range(self.num_bins):
            mask = (bins == b).to(magS.dtype)[None, None]  # (1,1,H,W)
            denom = mask.sum().clamp_min(1.0)
            ms = (magS * mask).sum(dim=(-2, -1)) / (denom + eps)
            mt = (magT * mask).sum(dim=(-2, -1)) / (denom + eps)
            loss = loss + torch.mean(torch.abs(ms - mt))
            valid_bins += 1

        return loss / max(1, valid_bins)


class LocalJacobianLoss(nn.Module):
    def __init__(self, sigma=1e-2):
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, S_fn, T_fn, x):
        eps = self.sigma * torch.randn_like(x)
        s1 = S_fn(x + eps) - S_fn(x)
        t1 = T_fn(x + eps) - T_fn(x)

        # Out-of-place NaN/Inf guards if either path can explode
        s1 = torch.nan_to_num(s1, nan=0.0, posinf=1e6, neginf=-1e6)
        t1 = torch.nan_to_num(t1, nan=0.0, posinf=1e6, neginf=-1e6)

        return torch.mean(torch.abs(s1 - t1))
