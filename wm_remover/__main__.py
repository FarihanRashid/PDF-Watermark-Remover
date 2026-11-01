import argparse
import sys
from dataclasses import replace


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    if len(sys.argv) == 1:
        from .gui import run
        run()
        return
    from .pipeline import PRESETS, process
    from .raster import MODELS
    B = argparse.BooleanOptionalAction
    ap = argparse.ArgumentParser(prog="wm_remover", description="Remove watermarks from PDFs (GPU accelerated).")
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--batch", nargs="+", metavar="PATH", help="PDF files and/or folders; results go to a subfolder next to each PDF")
    ap.add_argument("--out-name", default="cleaned", help="batch output subfolder name (default: cleaned)")
    ap.add_argument("--inspect", type=int, metavar="PAGE", help="describe what is in the bottom 12%% of PAGE (1-based) and exit")
    ap.add_argument("--preset", choices=list(PRESETS), default="balanced")
    ap.add_argument("--raster", choices=["auto", "always", "never"])
    ap.add_argument("--dpi", type=int, help="minimum render DPI while cleaning")
    ap.add_argument("--out-dpi", type=int, help="output DPI, e.g. 600 / 1200 / 1600 (0 = keep)")
    ap.add_argument("--upscale", choices=["bicubic", "smooth", "ai"])
    ap.add_argument("--ai-model", choices=list(MODELS))
    ap.add_argument("--erase", help="boxes x0,y0,x1,y1 as 0-1 fractions from top-left; ';' separates")
    ap.add_argument("--color-pages", choices=["keep", "grey"], help="vectorise mode: keep colour pages as colour rasters (default) or force grey vector")
    ap.add_argument("--max-upscale", type=float, help="safe mode: max output DPI as a multiple of the scan DPI (default 3)")
    ap.add_argument("--ink", type=float, help="luminance below this is always kept (tone cleaning)")
    ap.add_argument("--cut", type=float, help="luminance above this becomes paper white (tone cleaning)")
    ap.add_argument("--floor", type=float, help="darkest watermark the multi-page model corrects")
    for flag in ("structural", "multipage", "levels", "repeated-text", "repeat-dark", "png", "gray", "safe", "vectorize"):
        ap.add_argument(f"--{flag}", action=B, default=None)
    a = vars(ap.parse_args())
    if a["inspect"] and a["src"]:
        from .vector import inspect_page
        print(inspect_page(a["src"], a["inspect"] - 1))
        return
    if not a["batch"] and not (a["src"] and a["dst"]):
        ap.error("give SRC DST, or --batch PATH [PATH ...], or SRC --inspect PAGE")
    rename = {"erase": "erase_boxes", "ink": "ink_t", "cut": "wm_cut", "repeated_text": "remove_repeated_text",
              "png": "use_png"}
    over = {rename.get(k, k): v for k, v in a.items()
            if k not in ("src", "dst", "preset", "inspect", "batch", "out_name") and v is not None}
    o = replace(PRESETS[a["preset"]], **over)
    show = lambda f, m: print(f"{f:5.0%} {m}")
    if a["batch"]:
        from .pipeline import process_batch
        summ = process_batch(a["batch"], o, a["out_name"], show)
        for r in summ["results"]:
            print(r["status"].upper(), r["file"], r.get("output") or r.get("error") or "")
        return
    print(process(a["src"], a["dst"], o, show))


if __name__ == "__main__":
    main()
