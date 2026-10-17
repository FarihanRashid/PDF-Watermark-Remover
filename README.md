# PDF Watermark Remover (local, GPU-accelerated)

A Windows desktop tool that removes watermarks from PDFs **offline**, built for scanned or digital maths papers
(Cambridge O-Level style) where watermarks cross equations, graphs and diagrams. It prefers lossless, structural
removal and only touches pixels when it has to, with protection for dark ink so fractions, axes and numbers survive.

> **Use responsibly.** Only remove watermarks from material you own or are licensed to modify (your own teaching
> resources, papers you are permitted to adapt). Respect copyright and the terms of the source.

## Features
- **Stage 1 - structural (lossless):** edits the PDF content streams. Removes images/forms repeated on most pages
  (logos, crests, stamps), diagonal text objects, and repeated text lines (footers, URLs). Nothing is re-rendered, so
  equations stay vector and pixel-exact.
- **Stage 2 - pixel cleaning (GPU / CUDA):** for scans or flattened watermarks.
  - *Multi-page model:* a per-pixel high percentile across pages estimates "paper x watermark"; each page is divided
    by it (correct for alpha blending), so ink overlapping the watermark is scaled, not erased.
  - *Repeated dark marks:* whitens dark marks at the same pixel position on most pages (footers baked into scans).
  - *Single-page tone cleaning:* ink darker than a threshold is untouched; lighter tones fade to white.
- **Vectorise (B/W maths scans):** traces the cleaned page into true vector shapes (sub-pixel outlines from the
  anti-aliased scan, even-odd fills so the holes in 0/6/8/9 stay open). The result is tiny and stays sharp at any zoom,
  which no amount of raster upscaling can do. Light grey grid lines and shaded regions are kept as a separate grey layer.
- **Erase boxes:** blank fixed regions on every page (works on text, vectors and scanned image pixels).
- **Raster output:** output DPI, deterministic *smooth* upscaling (capped at 3x the scan's own DPI in safe mode),
  optional AI upscaling (Real-ESRGAN family via `spandrel`), greyscale and lossless PNG output.
- **Never freezes:** processing runs in a separate worker process; the window stays responsive and Cancel works.
- **Safety:** memory-aware banding/tiling, automatic retry on GPU out-of-memory, CPU fallback, RAM-based DPI limits,
  per-page error isolation, Cancel button, atomic output writing. See [Safety](#safety-features).
- **GUI with before/after preview**, presets, and a CLI.

## Requirements
- Windows 11 64-bit (Linux/macOS should work for the CPU path but are untested)
- NVIDIA GPU with CUDA support (developed for an RTX 2050, 4 GB) and a recent NVIDIA driver.
  The CUDA toolkit is **not** required: it ships inside the PyTorch wheel.
- Python 3.11 (3.10-3.12 should work)
- ~4 GB disk for PyTorch; 8 GB+ RAM recommended for 600+ DPI work

## Installation (PowerShell)
```powershell
# 1. Python 3.11 (skip if installed). With the Python install manager:
py install 3.11

# 2. In the project folder: virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass

# 3. Dependencies (PyTorch with CUDA first)
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 4. Verify the GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Do **not** use the "embeddable" Python zip: it has no `venv`/`tkinter` and ignores `PYTHONPATH`.

### Optional: AI upscaling
```powershell
pip install torchvision --index-url https://download.pytorch.org/whl/cu124
pip install spandrel
```
Models (5 MB or 67 MB) download from the Real-ESRGAN GitHub releases on first use into `models\`.

## Running
| Method | How |
|---|---|
| Double-click | `Launch.bat` (no console). `Launch_debug.bat` keeps a console open to show errors. |
| Desktop shortcut | Right-click `Create-Shortcut.ps1` > Run with PowerShell (once). |
| Command line (GUI) | `python -m wm_remover` |
| Command line (batch) | `python -m wm_remover in.pdf out.pdf [options]` |
| Standalone .exe (optional) | `.\build_exe.ps1` (3-5 GB because PyTorch + CUDA are bundled; untested on all setups) |

### GUI workflow
1. Pick the input PDF (the output name `_clean.pdf` is suggested).
2. Pick a **preset**, then click **Preview** on a page with a watermark over equations.
3. Tweak settings if needed, preview 2-3 different pages, then **Process whole PDF**.

### CLI examples
```powershell
python -m wm_remover in.pdf out.pdf                                   # balanced preset
python -m wm_remover in.pdf out.pdf --preset print-600
python -m wm_remover in.pdf out.pdf --repeat-dark --erase "0.25,0.94,0.75,0.99"
python -m wm_remover in.pdf out.pdf --out-dpi 1200 --upscale smooth --gray
python -m wm_remover in.pdf out.pdf --out-dpi 600 --upscale ai --ai-model "general-x4v3 (fast)"
```
Options: `--preset {balanced,print-600,vector,small-file}` `--[no-]vectorize` `--max-upscale F` `--raster {auto,always,never}` `--dpi N` (vector pages only; scans use their own DPI)
`--out-dpi N` `--upscale {bicubic,smooth,ai}` `--ai-model NAME` `--erase "x0,y0,x1,y1;..."` `--ink F` `--cut F`
`--floor F` `--inspect PAGE` and on/off pairs `--[no-]structural` `--[no-]multipage` `--[no-]levels` `--[no-]repeated-text`
`--[no-]repeat-dark` `--[no-]png` `--[no-]gray` `--[no-]safe`.

## Recommended settings
Start with the **balanced** preset and adjust by situation.

| Situation | Settings |
|---|---|
| Digital (text/vector) PDF, watermark is an object | Stage 1 on, **Rasterise = never**. Fully lossless. |
| Scanned / flattened, same watermark position on 3+ pages | balanced: Rasterise **auto**, **Multi-page model on**, tone cleaning **off** |
| Black footer / URL baked into a scan | Tick **Repeated dark marks**, or use **Erase boxes** e.g. `0.25,0.94,0.75,0.99` |
| Faint ghost of the watermark remains | Tick **Single-page tone cleaning**, raise **Watermark cut** 0.86 -> 0.92 |
| Thin lines / grey fills fade | Untick tone cleaning, or lower **Watermark cut**, raise **Ink protect** |
| Dark / opaque watermark survives | Lower **Multi-page floor** (0.45 -> 0.30); needs 3+ aligned same-size pages |
| Printing a raster scan | **print-600** - Output DPI 600 (limited to 3x the scan DPI), smooth, greyscale, PNG |
| **Sharp at any zoom, small file (B/W maths scans)** | **vector** preset: traces lines and text into true vector shapes |
| Smaller files | **small-file** (150 DPI, JPEG, greyscale) |
| Low-resolution scan (< 200 DPI) | Output DPI = 3-4x scan DPI, `ai` method, then **compare against the original** |

Notes: a raster scan only contains as much detail as its own DPI (e.g. 150 DPI = 1240x1754 px for A4). Upscaling
it to 1200 DPI makes a file ~60x larger with no real gain - zooming is just as blurry. **Vectorise** is the way to
get sharpness at any zoom. Output DPI cannot add detail the scan never had - it only makes edges smoother. Use greyscale for black &
white papers (3x smaller). AI upscalers can change digits and thin symbols; `smooth` never invents detail.

## Safety features
- **VRAM-aware processing:** pixel work runs in bands/tiles sized from free GPU memory. On a CUDA out-of-memory
  error the band/tile is halved and retried; if it still fails, cleaning/resizing finishes on the CPU and AI
  upscaling falls back to `smooth`.
- **RAM guard (Safe mode):** output DPI is automatically reduced so the page image fits in free RAM; cleaning is
  capped at 600 DPI and 120 megapixels per page; multi-page analysis uses fewer sample pages when RAM is tight.
- **Per-page isolation:** if a page fails, the original page is kept and the problem is listed at the end - the
  rest of the document is still processed.
- **Cancel button** and **atomic writes** (`.part` file renamed on success): no half-written or clobbered output.
- Refuses to overwrite the input file and checks for at least 500 MB of free disk space.
- Large arrays are held in system RAM, not VRAM, so the GPU only holds one band at a time.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No module named wm_remover` | Run from the project folder with the venv active; use `python -m wm_remover`. Don't use the embeddable Python. |
| `CUDA Available: False` | Update the NVIDIA driver; reinstall torch with the `cu124` index URL; check `nvidia-smi`. |
| GPU shows no load in Task Manager | Open Performance > GPU and switch a graph to **Cuda** (or Compute_0); or run `nvidia-smi -l 1`. Most time is spent on CPU rendering/encoding; AI upscaling shows the heaviest GPU load. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process Bypass` |
| `curl -L` fails in PowerShell | `curl` is an alias of `Invoke-WebRequest`; use `Invoke-WebRequest -Uri ... -OutFile ...` |
| A footer / URL is still there | 1) `python -m wm_remover file.pdf --inspect 3` (or the **Inspect footer** button) lists what the bottom 12% of page 3 is made of (text, image, vector paths, annotation, link, layer). 2) Use **Erase boxes** with the fractions it prints, e.g. `0.30,0.955,0.70,0.99`; boxes remove text, vector outlines, image pixels, annotations and links. 3) For scans you can instead tick **Repeated dark marks** (needs 3+ aligned pages and Rasterise = auto/always). |
| Window shows "Not Responding" | Fixed in v8: work runs in a separate process. Update, and avoid giant outputs (huge PNG pages make PyMuPDF's final save slow). |
| Output file is 40x bigger than the input | You upscaled a low-DPI scan. Use the **vector** preset, or Output DPI <= 3x the scan DPI, greyscale, JPEG. |
| Zooming in is still blurry | Raster pages cannot get sharper than their source DPI. Use **vector**. |
| Output looks soft | Output quality: PNG on, Output DPI >= scan DPI, keep tone cleaning off |
| Output file is huge | Greyscale on, lower Output DPI, or JPEG (small-file preset) |
| AI mode does nothing | `pip install spandrel torchvision` (see above); the app falls back to `smooth` and reports why |

## Known limits
- A single scanned page with an **opaque** watermark over equations has lost that information. Only a learned
  inpainter (e.g. LaMa) could guess it back, and it may guess wrong on maths. Not included.
- **Vectorise** outputs greyscale outlines and text is no longer selectable (it becomes shapes); very faint marks
  below the light-layer level disappear. Pages that contain colour (coloured graphs, photos) are detected and, with
  *Pages with colour = keep* (default), stay colour rasters at the scan's own DPI (JPEG); *grey* forces vector. Check the preview, especially thin graph lines.
- The multi-page model needs 3+ same-size pages with the watermark at the same pixel position.
- The repeated-text/mark removers delete *any* content repeated in the same place on most pages - check the preview.
- Erase boxes use the page as displayed; pages with unusual rotation may need coordinates adjusted.
- The preview does not apply output DPI or upscaling; test those on a one-page PDF.
- The GPU code paths were developed without access to an RTX 2050 test rig; please report issues.

## Project layout
```
wm_remover/
  __main__.py   CLI / GUI entry point
  gui.py        tkinter interface, presets, preview, cancel
  pipeline.py   options, presets, page loop, safety guards, worker process
  vector.py     structural removal (content streams, repeated text, erase boxes) + vector tracing
  raster.py     GPU cleaning, resampling, AI upscaling, OOM handling
Launch.bat  Launch_debug.bat  Create-Shortcut.ps1  build_exe.ps1  run.py
requirements.txt  models/ (downloaded AI weights, git-ignored)
```

