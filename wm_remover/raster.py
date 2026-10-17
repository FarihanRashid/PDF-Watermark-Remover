"""Pixel-domain cleaning and resampling on the GPU (PyTorch/CUDA), with memory safety.

Model: observed = (1 - a) * clean + a * W  (watermark W blended with opacity a).
 * flat_field(): divides by an across-pages estimate of 'paper x watermark'.
 * levels_clean(): single-page fallback; ink stays exact, light tones fade to white.
 * estimate_repeated_dark(): dark marks at the same pixel position on most pages (scan footers).
 * resample(): banded bicubic resize (+ optional unsharp) so 1200+ DPI fits in 4 GB of VRAM.
 * ai_upscale(): optional Real-ESRGAN-family models via `spandrel` (tile based).

Safety: all pixel work runs in bands/tiles sized from the free VRAM. On a CUDA out-of-memory error the
band/tile is halved and retried; if that still fails the work continues on the CPU (cleaning/resizing)
or the caller falls back to a cheaper method (AI upscaling). Large arrays live in system RAM, not VRAM.
"""
import gc
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

_force_cpu = False


def force_cpu(flag=True):
    global _force_cpu
    _force_cpu = flag


def fell_back_to_cpu():
    return _force_cpu and torch.cuda.is_available()


def device():
    return torch.device("cuda" if torch.cuda.is_available() and not _force_cpu else "cpu")


def gpu_info(mem=False):
    if torch.cuda.is_available():
        if mem:                                  # only in the worker: this creates a CUDA context
            free, total = torch.cuda.mem_get_info()
            return f"CUDA: {torch.cuda.get_device_name(0)} ({free / 2**30:.1f} of {total / 2**30:.1f} GB free)"
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    return "CUDA not available - running on CPU"


def free_vram():
    if device().type != "cuda":
        return None
    try:
        return torch.cuda.mem_get_info()[0]
    except Exception:
        return None


def release():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _rows(width, bytes_per_px, lo, hi, share=0.35):
    """How many rows fit into `share` of the free VRAM."""
    free = free_vram()
    if free is None:
        return hi
    return int(max(lo, min(hi, free * share / max(1.0, width * bytes_per_px))))


def to_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device()).permute(2, 0, 1).float().div_(255.0)


def to_numpy(t):
    return t.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def _paper(x):
    return torch.quantile(x[..., ::4, ::4].flatten()[:15_000_000], 0.98).clamp(min=1e-3)


def paper_of_bg(Wbg):
    return torch.stack([_paper(Wbg[c]) for c in range(3)]).view(3, 1, 1)


def page_paper(arr):
    sub = arr[::4, ::4].astype(np.float32)
    lum = (0.299 * sub[..., 0] + 0.587 * sub[..., 1] + 0.114 * sub[..., 2]) / 255.0
    return max(float(np.percentile(lum, 98)), 1e-3)


def _try_bands(fn, bands):
    """Run fn(band) with decreasing band sizes while CUDA runs out of memory."""
    last = None
    for b in bands:
        oom = False
        try:
            return fn(b)
        except Exception as e:
            if not is_oom(e):
                raise
            last, oom = e, True
        if oom:
            release()
    raise last


def _stack(pages, y, band):
    return torch.from_numpy(np.stack([p[y:y + band] for p in pages])).to(device()).permute(0, 3, 1, 2).float()


@torch.no_grad()
def estimate_background(pages, percentile=0.85):
    """Per-pixel high percentile across same-size pages -> 3xHxW float CPU tensor in [0,1]."""
    n = len(pages)
    k = max(1, min(n, int(np.ceil(percentile * n))))
    H, W = pages[0].shape[:2]

    def run(band):
        out = torch.empty(3, H, W)
        for y in range(0, H, band):
            out[:, y:y + band] = (_stack(pages, y, band).kthvalue(k, dim=0).values / 255.0).cpu()
        return out
    return _try_bands(run, (192, 96, 48, 24))


@torch.no_grad()
def estimate_repeated_dark(pages, frac=0.6, thr=0.5, grow=2):
    """Mask (1xHxW, CPU) of pixels dark at the same position on >= frac of the pages."""
    n = len(pages)
    k = max(1, min(n, int(np.ceil(frac * n))))
    H, W = pages[0].shape[:2]

    def run(band):
        m = torch.zeros(1, H, W)
        for y in range(0, H, band):
            lum = _stack(pages, y, band).mean(1) / 255.0
            m[0, y:y + band] = (lum.kthvalue(k, dim=0).values < thr).float().cpu()
        return m
    m = _try_bands(run, (192, 96, 48, 24))
    if grow:
        m = F.max_pool2d(m.unsqueeze(0), 2 * grow + 1, 1, grow)[0]
    return m


@torch.no_grad()
def has_watermark_evidence(Wbg, floor, min_fraction=0.002):
    lum = (Wbg / paper_of_bg(Wbg)).mean(0)
    return ((lum < 0.97) & (lum > floor)).float().mean().item() > min_fraction


@torch.no_grad()
def flat_field(img, Wband, paper, floor):
    wn = (Wband / paper).clamp(max=1.0)
    wn = torch.where(wn < floor, torch.ones_like(wn), wn)   # repeated dark content (headers) is NOT watermark
    return (img / (paper * wn)).clamp(0, 1)


@torch.no_grad()
def levels_clean(img, ink_t, wm_cut, paper_l):
    lum = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
    t = (((lum / paper_l) - ink_t) / max(wm_cut - ink_t, 1e-3)).clamp(0, 1)   # 0 = ink, 1 = watermark/paper
    return (img / paper_l).clamp(0, 1).mul(1 - t).add(t)


@torch.no_grad()
def clean_banded(arr, W, M, paperW, floor, levels, ink_t, wm_cut):
    H, width = arr.shape[:2]
    out = np.empty_like(arr)
    pl = page_paper(arr) if levels else 1.0
    band = _rows(width, 144, 16, 512)
    y = 0
    while y < H:
        b = min(band, H - y)
        oom = False
        try:
            dev = device()
            t = to_tensor(arr[y:y + b])
            if W is not None:
                t = flat_field(t, W[:, y:y + b].to(dev), paperW.to(dev), floor)
            if M is not None:
                t = torch.where(M[:, y:y + b].to(dev) > 0, torch.ones_like(t), t)
            if levels:
                t = levels_clean(t, ink_t, wm_cut, pl)
            out[y:y + b] = to_numpy(t)
            y += b
        except Exception as e:
            if not is_oom(e):
                raise
            oom = True
        if oom:
            t = None
            release()
            if band > 16:
                band = max(16, band // 2)
            elif device().type == "cuda":
                force_cpu(True)          # last resort: finish on the CPU instead of crashing
            else:
                raise MemoryError("Out of memory while cleaning a page")
    return out


# ---------------------------------------------------------------- resampling / upscaling
def _blur(t, sigma):
    r = max(1, int(np.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=t.device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = t.shape[1]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


@torch.no_grad()
def resample(arr, scale, sharpen=0.0, pad=8):
    """Bicubic resize of an HxWxC uint8 array by `scale`; optional unsharp mask restores edge crispness.
    Deterministic (no generative step), so digits and thin strokes are never invented or changed."""
    H, W, C = arr.shape
    oh, ow = int(round(H * scale)), int(round(W * scale))
    out = np.empty((oh, ow, C), np.uint8)
    band = _rows(ow, 72 * max(scale, 1.0), 8, 256)
    y0 = 0
    while y0 < H:
        y1 = min(H, y0 + band)
        a0, a1 = max(0, y0 - pad), min(H, y1 + pad)
        oom = False
        try:
            t = to_tensor(arr[a0:a1]).unsqueeze(0)
            r0, r1 = int(round(a0 * scale)), int(round(a1 * scale))
            up = F.interpolate(t, size=(max(1, r1 - r0), ow), mode="bicubic", align_corners=False,
                               antialias=scale < 1)
            if sharpen > 0 and scale > 1:
                up = up + sharpen * (up - _blur(up, 0.35 * scale))
            o0 = int(round(y0 * scale))
            o1 = oh if y1 == H else int(round(y1 * scale))
            out[o0:o1] = to_numpy(up[0, :, o0 - r0:o1 - r0, :])
            y0 = y1
        except Exception as e:
            if not is_oom(e):
                raise
            oom = True
        if oom:
            t = up = None
            release()
            if band > 8:
                band = max(8, band // 2)
            elif device().type == "cuda":
                force_cpu(True)
            else:
                raise MemoryError("Out of memory while resizing a page")
    return out


MODELS = {
    "general-x4v3 (fast)": ("realesr-general-x4v3.pth",
                            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
    "x4plus (best, slower)": ("RealESRGAN_x4plus.pth",
                              "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
}
_loaded = {}


def load_ai(name):
    if name in _loaded:
        return _loaded[name]
    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError as e:
        raise RuntimeError("AI upscaling needs the 'spandrel' package (see README)") from e
    fn, url = MODELS[name]
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fn)
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    model = ModelLoader().load_from_file(path)
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError("Unsupported model file")
    model.to(device()).eval()
    _loaded[name] = model
    return model


def _pick_tile(model):
    free = free_vram()
    tile = 128 if free is None else (256 if free > 5e9 else 160 if free > 2.5e9 else 96)
    try:
        if sum(p.numel() for p in model.model.parameters()) > 5e6:   # heavy RRDB model
            tile //= 2
    except Exception:
        pass
    return max(48, tile)


@torch.no_grad()
def _ai_pass(arr, model, tile, pad=16):
    s = int(model.scale)
    gray = arr.shape[2] == 1
    src = np.repeat(arr, 3, axis=2) if gray else arr
    H, W = src.shape[:2]
    out = np.empty((H * s, W * s, 3), np.uint8)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y1, x1 = min(H, y + tile), min(W, x + tile)
            a0, b0, a1, b1 = max(0, y - pad), max(0, x - pad), min(H, y1 + pad), min(W, x1 + pad)
            o = to_numpy(model(to_tensor(src[a0:a1, b0:b1]).unsqueeze(0))[0])
            cy, cx = (y - a0) * s, (x - b0) * s
            out[y * s:y1 * s, x * s:x1 * s] = o[cy:cy + (y1 - y) * s, cx:cx + (x1 - x) * s]
    if gray:
        out = (out.astype(np.uint16).sum(2) // 3).astype(np.uint8)[..., None]
    return out, s


def ai_upscale(arr, model):
    """Tile-based super-resolution; halves the tile on CUDA OOM. Raises MemoryError if even 48 px fails."""
    tile = _pick_tile(model)
    while True:
        oom = False
        try:
            return _ai_pass(arr, model, tile)
        except Exception as e:
            if not is_oom(e):
                raise
            oom = True
        if oom:
            release()
            if tile <= 48:
                raise MemoryError("GPU out of memory even with the smallest AI tile")
            tile = max(48, tile // 2)
