"""Stage 1 - structural removal (lossless). Edits the PDF content streams directly so the
maths underneath is never touched. Handles:
  * images / form XObjects that repeat identically on most pages (logos, seals, stamps)
  * rotated (diagonal) text objects, incl. text inside a rotated `q ... cm ... Q` block
"""
import hashlib
import math
import re
from collections import Counter

import numpy as np
import pymupdf as fitz

NUM = rb"(-?\d*\.?\d+)"
ROT = re.compile(rb"\s+".join([NUM] * 6) + rb"\s+(Tm|cm)\b")
TEXTOP = re.compile(rb"\b(?:Tj|TJ)\b")


def _digest(doc, xref, image):
    try:
        data = doc.extract_image(xref)["image"] if image else (doc.xref_stream(xref) or b"")
    except Exception:
        return None
    return hashlib.sha1(data).hexdigest() if data else None


def _is_rot(m):
    a, b = float(m.group(1)), float(m.group(2))
    s = abs(b) / (math.hypot(a, b) or 1.0)
    return 0.03 < s < 0.97          # not axis aligned (keeps 0/90/180/270 deg axis labels)


def strip_rotated_text(s):
    """Return (new_stream, removed_count)."""
    removed = 0

    def bt(m):
        nonlocal removed
        seg = m.group(0)
        if TEXTOP.search(seg) and any(r.group(7) == b"Tm" and _is_rot(r) for r in ROT.finditer(seg)):
            removed += 1
            return b""
        return seg

    s = re.sub(rb"\bBT\b.*?\bET\b", bt, s, flags=re.S)

    # q ... Q blocks that set a rotated CTM and contain text
    pairs, stack = [], []
    for m in re.finditer(rb"(?<![\w/(])([qQ])(?![\w)])", s):
        if m.group(1) == b"q":
            stack.append(m.start())
        elif stack:
            pairs.append((stack.pop(), m.end()))

    def cand(p):
        seg = s[p[0]:p[1]]
        return (b"BT" in s[p[0]:p[1]] and TEXTOP.search(seg)
                and any(r.group(7) == b"cm" and _is_rot(r) for r in ROT.finditer(seg)))

    cands = [p for p in pairs if cand(p)]
    minimal = [p for p in cands if not any(q != p and p[0] < q[0] and q[1] < p[1] for q in cands)]
    for a, b in sorted(minimal, reverse=True):
        s = s[:a] + s[b:]
        removed += 1
    return s, removed


def strip_repeated_text(doc, ratio, min_len=4, tol=3.0):
    """Remove text lines (footers, URLs, copyright lines) that repeat with the same wording at the
    same position on >= ratio of the pages. Page numbers / question numbers differ per page, so they stay."""
    n = len(doc)
    per_page, counts = [], Counter()
    for pno in range(n):
        keys = {}
        for b in doc[pno].get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                txt = "".join(s["text"] for s in ln["spans"]).strip()
                if len(txt) < min_len:
                    continue
                x0, y0 = ln["bbox"][0], ln["bbox"][1]
                keys[(txt, round(x0 / tol), round(y0 / tol))] = fitz.Rect(ln["bbox"])
        per_page.append(keys)
        counts.update(keys.keys())
    need = max(3, math.ceil(ratio * n))
    removed = 0
    for pno, keys in enumerate(per_page):
        rects = [r for k, r in keys.items() if counts[k] >= need]
        if not rects:
            continue
        page = doc[pno]
        for r in rects:
            page.add_redact_annot(r, fill=False)
        page.apply_redactions(images=getattr(fitz, "PDF_REDACT_IMAGE_NONE", 0),
                              graphics=getattr(fitz, "PDF_REDACT_LINE_ART_NONE", 0))
        removed += len(rects)
    return removed


def parse_boxes(text):
    """'x0,y0,x1,y1; ...' as fractions of page width/height measured from the top-left."""
    boxes = []
    for part in (text or "").split(";"):
        if part.strip():
            v = [float(x) for x in part.split(",")]
            if len(v) != 4:
                raise ValueError("Each erase box needs 4 numbers: x0,y0,x1,y1")
            boxes.append(tuple(v))
    return boxes


def erase_boxes(doc, boxes):
    """Blank fixed regions on every page: text, vector outlines, image pixels, annotations and links."""
    for page in doc:
        r = page.rect
        rects = [fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height,
                           r.x0 + x1 * r.width, r.y0 + y1 * r.height) for x0, y0, x1, y1 in boxes]
        for a in list(page.annots() or []):          # footers added as stamp/free-text annotations
            if any(a.rect.intersects(q) for q in rects):
                page.delete_annot(a)
        for ln in page.get_links():
            if any(fitz.Rect(ln["from"]).intersects(q) for q in rects):
                page.delete_link(ln)
        for q in rects:
            page.add_redact_annot(q, fill=(1, 1, 1))
        page.apply_redactions(images=getattr(fitz, "PDF_REDACT_IMAGE_PIXELS", 2),
                              graphics=getattr(fitz, "PDF_REDACT_LINE_ART_REMOVE_IF_COVERED", 1))


def strip_overlay_annotations(doc):
    """Remove watermark / stamp / free-text annotations (a common way sites add overlays)."""
    kinds = {getattr(fitz, n, -1) for n in ("PDF_ANNOT_WATERMARK", "PDF_ANNOT_STAMP", "PDF_ANNOT_FREE_TEXT")}
    removed = 0
    for page in doc:
        for a in list(page.annots() or []):
            if a.type[0] in kinds:
                page.delete_annot(a)
                removed += 1
    return removed


def inspect_page(src, pno, region=(0.0, 0.88, 1.0, 1.0)):
    """Describe what lives in a page region (default: bottom 12%) so a stubborn mark can be diagnosed."""
    doc = fitz.open(src)
    page = doc[pno]
    r = page.rect
    box = fitz.Rect(r.x0 + region[0] * r.width, r.y0 + region[1] * r.height,
                    r.x0 + region[2] * r.width, r.y0 + region[3] * r.height)
    cov = sum((fitz.Rect(i["bbox"]) & r).get_area() for i in page.get_image_info()) / (r.get_area() or 1)
    out = [f"Page {pno + 1}: {r.width:.0f}x{r.height:.0f} pt, rotation {page.rotation}, "
           f"image coverage {cov:.0%} ({'scanned-looking' if cov > 0.5 else 'digital'}), "
           f"region fractions {region}"]
    for i in page.get_image_info(xrefs=True):
        if fitz.Rect(i["bbox"]).intersects(box):
            b = fitz.Rect(i["bbox"])
            out.append(f"  image xref={i.get('xref')} {i['width']}x{i['height']}px at "
                       f"x {(b.x0 - r.x0) / r.width:.2f}-{(b.x1 - r.x0) / r.width:.2f}, "
                       f"y {(b.y0 - r.y0) / r.height:.2f}-{(b.y1 - r.y0) / r.height:.2f} of page")
    for blk in page.get_text("dict", clip=box)["blocks"]:
        if blk.get("type") == 0:
            for ln in blk["lines"]:
                t = "".join(s["text"] for s in ln["spans"]).strip()
                if t:
                    b = fitz.Rect(ln["bbox"])
                    out.append(f"  text '{t[:60]}' font={ln['spans'][0]['font']} "
                               f"x {(b.x0 - r.x0) / r.width:.2f}-{(b.x1 - r.x0) / r.width:.2f}, "
                               f"y {(b.y0 - r.y0) / r.height:.2f}-{(b.y1 - r.y0) / r.height:.2f}")
    n = sum(1 for d in page.get_drawings() if fitz.Rect(d["rect"]).intersects(box))
    out.append(f"  vector paths in region: {n}")
    for a in list(page.annots() or []):
        if a.rect.intersects(box):
            out.append(f"  annotation {a.type[1]} at {tuple(round(v) for v in a.rect)}")
    for ln in page.get_links():
        if fitz.Rect(ln["from"]).intersects(box):
            out.append(f"  link {ln.get('uri') or ln.get('page')}")
    try:
        layers = doc.get_ocgs()
        out.append(f"  optional-content layers in file: {len(layers)} "
                   f"{[v.get('name') for v in layers.values()][:5]}")
    except Exception:
        pass
    if len(out) == 1:
        out.append("  (nothing found in this region)")
    return "\n".join(out)


def strip(doc, o):
    n = len(doc)
    report = {"repeated_objects": 0, "rotated_text": 0, "repeated_text": 0}
    if getattr(o, "remove_repeated_text", False):
        report["repeated_text"] = strip_repeated_text(doc, o.repeat_ratio)
        report["overlay_annotations"] = strip_overlay_annotations(doc)
    per_page, counts = [], Counter()
    for pno in range(n):
        items = []
        if o.remove_images:
            for img in doc.get_page_images(pno, full=True):
                items.append((img[7], _digest(doc, img[0], True)))
        if o.remove_forms:
            for xref, name, _inv, _bbox in doc.get_page_xobjects(pno):
                items.append((name, _digest(doc, xref, False)))
        per_page.append(items)
        for d in {d for _, d in items if d}:
            counts[d] += 1
    need = max(3, math.ceil(o.repeat_ratio * n))

    for pno, items in enumerate(per_page):
        names = {nm for nm, d in items if d and nm and counts[d] >= need}
        if not names and not o.remove_rotated_text:
            continue
        page = doc[pno]
        page.clean_contents()
        for cx in page.get_contents():
            s = doc.xref_stream(cx)
            if not s:
                continue
            new = s
            for nm in names:
                new, k = re.subn(rb"/" + re.escape(nm.encode()) + rb"\s+Do\b", b"", new)
                report["repeated_objects"] += k
            if o.remove_rotated_text:
                new, k = strip_rotated_text(new)
                report["rotated_text"] += k
            if new != s:
                doc.update_stream(cx, new)
    return report


def trace_page(page, up, scale, dpi, ink=0.5, grey=0.86, min_size=0.8, tol=0.3):
    """Turn a cleaned greyscale page image into true vector shapes on `page`.

    `up` is the (optionally upsampled) HxW uint8 image, `scale` = up size / source size, `dpi` = source DPI.
    Marching squares on the anti-aliased image gives sub-pixel outlines, so edges stay smooth at any zoom.
    Two layers: light grey (grid lines, shaded regions) first, then black ink on top. Even-odd fill keeps
    holes (the inside of 0, 6, 8, 9 ...). Returns the number of traced shapes."""
    try:
        from skimage import measure
    except ImportError as e:
        raise RuntimeError("Vectorising needs scikit-image: pip install scikit-image") from e
    img = up.astype(np.float32) / 255.0
    paper = max(float(np.percentile(img[::4, ::4], 98)), 1e-3)
    img = np.pad(np.clip(img / paper, 0, 1), 1, constant_values=1.0)     # white border closes every outline
    k = 72.0 / dpi / scale                                               # upsampled pixel -> PDF points

    def layer(level, colour):
        sh, n = page.new_shape(), 0
        for c in measure.find_contours(img, level):
            if len(c) < 4:
                continue
            c = measure.approximate_polygon(c, tol)
            if len(c) < 4 or (c.max(0) - c.min(0)).max() < min_size * scale:
                continue
            sh.draw_polyline([fitz.Point(round((x - 0.5) * k, 2), round((y - 0.5) * k, 2)) for y, x in c])
            n += 1
        if n:
            sh.finish(fill=colour, color=None, even_odd=True, closePath=True)
            sh.commit()
        return n

    # fill colour for the light layer: flat interior of shaded regions if there are any, else the typical
    # value of thin light lines (never the anti-aliasing fringe around ink, which would look like a halo)
    from scipy import ndimage
    band = (img > ink) & (img < grey)
    flat = ndimage.binary_erosion(band, iterations=2)
    if flat.sum() > 200:
        g = float(np.median(img[flat]))
    elif band.sum() > 50:
        g = float(np.percentile(img[band], 25))
    else:
        g = 0.8
    g = float(np.clip(g, 0.55, 0.86))
    total = layer(grey, (g, g, g))
    total += layer(ink, (0, 0, 0))
    return total
