import io
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
import pymupdf as fitz
from PIL import Image

from . import raster, vector

Image.MAX_IMAGE_PIXELS = None

MAX_CLEAN_DPI = 600
MAX_CLEAN_PIXELS = 120e6       # per page, while cleaning (safe mode)


@dataclass
class Options:
    structural: bool = True
    remove_images: bool = True
    remove_forms: bool = True
    remove_rotated_text: bool = True
    remove_repeated_text: bool = True
    erase_boxes: str = ""          # e.g. "0.2,0.94,0.8,0.99" (fractions, top-left origin); separate with ;
    repeat_ratio: float = 0.6
    raster: str = "auto"           # auto (scanned-looking pages) | always | never
    dpi: int = 200                 # render DPI for pages without a scanned image (scans use their own DPI)
    out_dpi: int = 0               # 0 = keep render DPI; e.g. 600 / 1200 / 1600 for print
    upscale: str = "smooth"        # bicubic | smooth (bicubic + edge sharpen) | ai (Real-ESRGAN via spandrel)
    ai_model: str = "general-x4v3 (fast)"
    sharpen: float = 0.8
    gray: bool = False             # greyscale output (3x smaller, fine for black & white papers)
    multipage: bool = True
    floor: float = 0.45
    levels: bool = False
    ink_t: float = 0.60
    wm_cut: float = 0.86
    jpeg_quality: int = 95
    use_png: bool = False
    repeat_dark: bool = False      # whiten dark marks repeated at the same pixel position (footers on scans)
    match_native: bool = True      # render at the scan's native resolution (sharper)
    safe: bool = True              # memory guards: clamp DPI to available RAM, limit page pixels
    max_upscale: float = 3.0       # safe mode: never upscale a scan beyond this multiple of its own DPI
    color_pages: str = "keep"      # vectorise mode: 'keep' pages that contain colour as colour rasters | 'grey' force vector
    vectorize: bool = False        # trace cleaned B/W pages into true vector shapes (tiny, sharp at any zoom)
    trace_ink: float = 0.5         # vectorise: luminance level of the black outline
    trace_grey: float = 0.86       # vectorise: level of the light layer (grid lines, shaded regions)


_B = Options(use_png=True)
PRESETS = {
    "balanced": _B,
    "print-600": replace(_B, out_dpi=600, gray=True),
    "vector": replace(_B, vectorize=True),
    "small-file": replace(_B, dpi=150, use_png=False, jpeg_quality=85, gray=True),
}


def avail_ram():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MS()
        s.dwLength = ctypes.sizeof(s)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return int(s.ullAvailPhys)
    except Exception:
        return None


def fit_out_dpi(rect, want, channels):
    """Largest output DPI <= want whose page image fits comfortably in free RAM."""
    w, h = rect.width / 72, rect.height / 72
    dpi = min(float(want), 60000 / max(w, h))          # also keeps JPEG/PNG dimensions sane
    avail = avail_ram()
    if avail:
        maxpx = avail * 0.55 / (channels * 3.5)         # array + encoder + PDF copy
        dpi = min(dpi, (maxpx / (w * h)) ** 0.5)
    return int(dpi)


def is_scanned(page, thr=0.5):
    area = page.rect.get_area() or 1
    cov = sum((fitz.Rect(i["bbox"]) & page.rect).get_area() for i in page.get_image_info())
    return cov / area > thr


class Cleaner:
    def __init__(self, doc, o, targets):
        self.doc, self.o = doc, o
        self.groups = defaultdict(list)
        self.warnings = []
        for i in targets:
            r = doc[i].rect
            self.groups[(round(r.width), round(r.height))].append(i)
        self.bg = {}

    def dpi_for(self, i):
        page, dpi = self.doc[i], self.o.dpi
        try:
            infos = page.get_image_info()
        except Exception:
            infos = []
        if infos and self.o.match_native:
            b = max(infos, key=lambda d: d["width"] * d["height"])
            w = b["bbox"][2] - b["bbox"][0]
            if w > 0:
                dpi = b["width"] * 72 / w          # scans: render at the image's own DPI (no resampling blur)
        elif not infos and self.o.out_dpi:
            dpi = max(dpi, self.o.out_dpi)          # vector page: render straight at the wanted resolution
        dpi = min(dpi, MAX_CLEAN_DPI)
        r = page.rect
        if self.o.safe:
            area_in = (r.width / 72) * (r.height / 72)
            dpi = min(dpi, (MAX_CLEAN_PIXELS / max(area_in, 1e-6)) ** 0.5)
        dpi = int(round(dpi))
        if self.o.out_dpi and self.o.out_dpi < dpi:
            dpi = max(72, self.o.out_dpi)
        return max(72, dpi)

    def render(self, i):
        pix = self.doc[i].get_pixmap(dpi=self.dpi_for(i), colorspace=fitz.csRGB, alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()

    def _n_samples(self, i):
        r, d = self.doc[i].rect, self.dpi_for(i)
        page_bytes = (r.width / 72 * d) * (r.height / 72 * d) * 3
        avail = avail_ram()
        if self.o.safe and avail:
            return int(max(0, min(12, avail * 0.3 / page_bytes)))
        return 12

    def background(self, i):
        r = self.doc[i].rect
        key = (round(r.width), round(r.height))
        if key not in self.bg:
            idx, W, M, pw = self.groups[key], None, None, None
            if (self.o.multipage or self.o.repeat_dark) and len(idx) >= 3:
                try:
                    n = self._n_samples(idx[0])
                    if n < 3:
                        self.warnings.append("Multi-page analysis skipped: not enough free RAM at this DPI.")
                    else:
                        pick = [idx[j] for j in np.linspace(0, len(idx) - 1, min(n, len(idx))).round().astype(int)]
                        imgs = [self.render(p) for p in dict.fromkeys(pick)]
                        if len({im.shape for im in imgs}) == 1:
                            if self.o.multipage:
                                cand = raster.estimate_background(imgs)
                                if raster.has_watermark_evidence(cand, self.o.floor):
                                    W, pw = cand, raster.paper_of_bg(cand)
                            if self.o.repeat_dark:
                                M = raster.estimate_repeated_dark(imgs)
                        del imgs
                except Exception as e:
                    if not (raster.is_oom(e) or isinstance(e, MemoryError)):
                        raise
                    W = M = pw = None
                    raster.release()
                    self.warnings.append("Multi-page analysis skipped: ran out of memory.")
            self.bg[key] = (W, M, pw)
        return self.bg[key]

    def clean(self, i):
        arr = self.render(i)
        W, M, pw = self.background(i)
        if W is not None and tuple(W.shape[1:]) != arr.shape[:2]:
            W = None
        if M is not None and tuple(M.shape[1:]) != arr.shape[:2]:
            M = None
        return raster.clean_banded(arr, W, M, pw, self.o.floor, self.o.levels, self.o.ink_t, self.o.wm_cut)


def _gray(arr):
    g = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    return np.clip(g + 0.5, 0, 255).astype(np.uint8)[..., None]


def is_colourful(arr, chroma_thr=0.2, min_fraction=0.003):
    """True if a cleaned RGB page contains real colour (not just paper tint or JPEG fringes)."""
    if arr.shape[2] != 3:
        return False
    sub = arr[::2, ::2].astype(np.int16)
    chroma = sub.max(axis=2) - sub.min(axis=2)
    return float((chroma > chroma_thr * 255).mean()) > min_fraction


def _trace_scale(shape):
    """Upsample factor before tracing so the traced image is ~16 megapixels (smooth outlines, bounded time)."""
    return float(min(4.0, max(1.0, round(2 * (16e6 / (shape[0] * shape[1])) ** 0.5) / 2)))


def _trace(arr, dpi, o, page):
    g = _gray(arr)
    s = _trace_scale(g.shape)
    up = raster.resample(g, s, 0.0) if s > 1.01 else g
    return vector.trace_page(page, up[..., 0], up.shape[0] / g.shape[0], dpi, o.trace_ink, o.trace_grey)


def _finish(arr, dpi, o, warns):
    """Optional greyscale + upscaling to the requested output DPI."""
    if o.gray and arr.shape[2] == 3:
        arr = _gray(arr)
    if o.out_dpi and o.out_dpi > dpi * 1.02:
        method = o.upscale
        if method == "ai":
            avail = avail_ram()
            mid = arr.shape[0] * arr.shape[1] * 16 * arr.shape[2] * 2     # x4 model output + working copy
            if o.safe and avail and mid > avail * 0.5:
                warns.append("AI upscaling skipped on a page (not enough RAM); used 'smooth'.")
                method = "smooth"
            else:
                try:
                    arr, s = raster.ai_upscale(arr, raster.load_ai(o.ai_model))
                    dpi *= s
                    if abs(o.out_dpi - dpi) / dpi > 0.02:
                        arr = raster.resample(arr, o.out_dpi / dpi, 0.0)
                    return arr
                except (MemoryError, RuntimeError, ImportError) as e:
                    warns.append(f"AI upscaling failed on a page ({str(e)[:80]}); used 'smooth'.")
                    raster.release()
                    method = "smooth"
        arr = raster.resample(arr, o.out_dpi / dpi, o.sharpen if method == "smooth" else 0.0)
    return arr


def _encode(arr, o):
    buf = io.BytesIO()
    im = Image.fromarray(np.ascontiguousarray(arr[..., 0])) if arr.shape[2] == 1 else Image.fromarray(arr)
    if o.use_png:
        im.save(buf, "PNG", compress_level=6)
    else:
        im.save(buf, "JPEG", quality=o.jpeg_quality, subsampling=0)
    return buf.getvalue()


def _prepare(src, o):
    doc = fitz.open(src)
    report = vector.strip(doc, o) if o.structural else {}
    boxes = vector.parse_boxes(o.erase_boxes)
    if boxes:
        vector.erase_boxes(doc, boxes)
        report["erased_boxes"] = len(boxes)
    targets = [i for i in range(len(doc))
               if o.raster == "always" or (o.raster == "auto" and is_scanned(doc[i]))]
    return doc, targets, Cleaner(doc, o, targets), report


def process(src, dst, o, progress=lambda f, m: None, cancel=lambda: False):
    s, d = os.path.abspath(src), os.path.abspath(dst)
    if os.path.normcase(s) == os.path.normcase(d):
        raise ValueError("The output file must be different from the input file.")
    if shutil.disk_usage(os.path.dirname(d) or ".").free < 500 * 2 ** 20:
        raise OSError("Less than 500 MB free on the output drive.")
    doc, targets, cl, report = _prepare(src, o)
    tset, n = set(targets), len(doc)
    warns, failed, o_run = [], [], o
    if tset and o.out_dpi and o.upscale == "ai":
        progress(0.0, "Loading AI model (first use downloads it)...")
        try:
            raster.load_ai(o.ai_model)
        except Exception as e:
            warns.append(f"AI upscaling unavailable ({str(e)[:100]}); used 'smooth' instead.")
            o_run = replace(o, upscale="smooth")
    out, cancelled, colour_kept = fitz.open(), False, []
    for i in range(n):
        if cancel():
            cancelled = True
            break
        try:
            if i in tset:
                progress(i / n, f"page {i + 1}/{n}: cleaning")
                arr = cl.clean(i)
                if o.vectorize and o.color_pages == "keep" and is_colourful(arr):
                    colour_kept.append(i + 1)           # colour diagrams / photos: keep colour, skip tracing
                    pg = out.new_page(width=doc[i].rect.width, height=doc[i].rect.height)
                    pg.insert_image(pg.rect, stream=_encode(arr, replace(o, use_png=False)))
                    del arr
                    raster.release()
                    progress((i + 1) / n, f"page {i + 1}/{n} done (colour kept)")
                    continue
                if o.vectorize:
                    progress(i / n, f"page {i + 1}/{n}: tracing vectors")
                    pg = out.new_page(width=doc[i].rect.width, height=doc[i].rect.height)
                    _trace(arr, cl.dpi_for(i), o, pg)
                    del arr
                    raster.release()
                    progress((i + 1) / n, f"page {i + 1}/{n} done")
                    continue
                po = o_run
                if o.safe and o.out_dpi:
                    native = cl.dpi_for(i)
                    eff = fit_out_dpi(doc[i].rect, o.out_dpi, 1 if o.gray else 3)
                    cap = int(native * o.max_upscale)
                    if o.out_dpi > cap and eff > cap:
                        eff = cap
                        warns.append(f"Output DPI limited to {cap} ({o.max_upscale:g}x this scan's {native} DPI): "
                                     "more only adds file size, not detail. Use 'vectorise' for real sharpness.")
                    elif eff < o.out_dpi:
                        warns.append(f"Output DPI reduced from {o.out_dpi} to {eff} to fit in memory.")
                    po = replace(po, out_dpi=eff)
                progress(i / n, f"page {i + 1}/{n}: resampling / encoding")
                arr = _finish(arr, cl.dpi_for(i), po, warns)
                data = _encode(arr, po)
                del arr
                pg = out.new_page(width=doc[i].rect.width, height=doc[i].rect.height)
                pg.insert_image(pg.rect, stream=data)
                raster.release()
            else:
                out.insert_pdf(doc, from_page=i, to_page=i)
        except Exception as e:                      # one bad page must not lose the whole document
            failed.append(f"page {i + 1}: {str(e)[:100]}")
            raster.release()
            if len(out) > i:
                out.delete_page(i)
            try:
                out.insert_pdf(doc, from_page=i, to_page=i)
            except Exception:
                out.new_page(width=doc[i].rect.width, height=doc[i].rect.height)
        progress((i + 1) / n, f"page {i + 1}/{n} done")
    if colour_kept:
        warns.append("Pages with colour were kept as colour images instead of vectorised: "
                     + ", ".join(map(str, colour_kept[:30])) + ("..." if len(colour_kept) > 30 else ""))
    if cancelled:
        out.close()
        report.update(cancelled=True)
        return report
    tmp = d + ".part"
    out.save(tmp, garbage=4, deflate=True)
    out.close()
    os.replace(tmp, d)
    report.update(pages=n, rasterised_pages=len(targets), gpu=raster.gpu_info(mem=True),
                  warnings=list(dict.fromkeys(warns + cl.warnings)), failed_pages=failed)
    if raster.fell_back_to_cpu():
        report["warnings"].append("GPU memory ran out; part of the work finished on the CPU (slower).")
    return report


def worker(src, dst, o, q, cancel_ev):
    """Runs in a separate process so the window never freezes (PyMuPDF holds the GIL during big saves)."""
    try:
        rep = process(src, dst, o, lambda f, m: q.put(("p", f, m)), cancel_ev.is_set)
        q.put(("done", rep))
    except BaseException as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def preview_page(src, pno, o):
    """Returns (before, after) PIL images of one page. Output DPI / upscaling are skipped in the preview."""
    o = replace(o, out_dpi=0)
    orig = fitz.open(src)
    pix = orig[pno].get_pixmap(dpi=130, colorspace=fitz.csRGB)
    before = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc, targets, cl, _ = _prepare(src, o)
    if pno in targets:
        arr = cl.clean(pno)
        if o.vectorize and not (o.color_pages == "keep" and is_colourful(arr)):
            tmp = fitz.open()
            pg = tmp.new_page(width=doc[pno].rect.width, height=doc[pno].rect.height)
            _trace(arr, cl.dpi_for(pno), o, pg)
            pix = pg.get_pixmap(dpi=130, colorspace=fitz.csRGB)
            after = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        else:
            after = Image.fromarray(arr).convert("RGB")
            after.thumbnail(before.size, Image.LANCZOS)
    else:
        pix = doc[pno].get_pixmap(dpi=130, colorspace=fitz.csRGB)
        after = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return before, after
