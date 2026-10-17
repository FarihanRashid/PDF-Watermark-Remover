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
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
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
    if a["inspect"]:
        from .vector import inspect_page
        print(inspect_page(a["src"], a["inspect"] - 1))
        return
    if not a["dst"]:
        ap.error("dst is required unless --inspect is used")
    rename = {"erase": "erase_boxes", "ink": "ink_t", "cut": "wm_cut", "repeated_text": "remove_repeated_text",
              "png": "use_png"}
    over = {rename.get(k, k): v for k, v in a.items()
            if k not in ("src", "dst", "preset", "inspect") and v is not None}
    o = replace(PRESETS[a["preset"]], **over)
    rep = process(a["src"], a["dst"], o, lambda f, m: print(f"{f:5.0%} {m}"))
    print(rep)


if __name__ == "__main__":
    main()
