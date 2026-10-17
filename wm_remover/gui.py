import multiprocessing as mp
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pymupdf as fitz
from PIL import Image, ImageTk

from .pipeline import PRESETS, Options, preview_page, worker
from .raster import MODELS, gpu_info


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Watermark Remover (GPU)")
        self.geometry("1200x900")
        self.src, self.dst, self.erase = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.proc, self.q, self.cancel_mp = None, None, None
        B = lambda v: tk.BooleanVar(value=v)
        S = tk.StringVar
        self.v = {
            "structural": B(True), "images": B(True), "forms": B(True), "rotated": B(True), "repeated": B(True),
            "multipage": B(True), "levels": B(False), "repdark": B(False), "png": B(True), "gray": B(False),
            "safe": B(True), "vector": B(False), "colorpages": S(value="keep"), "preset": S(value="balanced"),
            "raster": S(value="auto"), "upscale": S(value="smooth"), "aimodel": S(value=list(MODELS)[0]),
            "dpi": S(value="200"), "outdpi": S(value="0"),
            "ink": tk.DoubleVar(value=0.60), "cut": tk.DoubleVar(value=0.86), "floor": tk.DoubleVar(value=0.45),
            "page": tk.IntVar(value=1),
        }
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self):
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
        self.destroy()

    def _build(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        ttk.Label(left, text=gpu_info(), foreground="#0a7").pack(anchor="w", pady=(0, 8))
        for label, var, save in (("Input PDF", self.src, False), ("Output PDF", self.dst, True)):
            ttk.Label(left, text=label).pack(anchor="w")
            row = ttk.Frame(left); row.pack(fill="x", pady=2)
            ttk.Entry(row, textvariable=var, width=36).pack(side="left")
            ttk.Button(row, text="...", width=3, command=lambda v=var, s=save: self._pick(v, s)).pack(side="left")

        row = ttk.Frame(left); row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Preset").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.v["preset"], values=list(PRESETS), width=14, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.apply_preset())

        nb = ttk.Notebook(left); nb.pack(fill="x", pady=6)
        t1, t2 = ttk.Frame(nb, padding=6), ttk.Frame(nb, padding=6)
        nb.add(t1, text="Remove"); nb.add(t2, text="Output quality")

        box = ttk.LabelFrame(t1, text="Stage 1 - structural (lossless)", padding=6); box.pack(fill="x")
        for t, k in (("Enable", "structural"), ("Repeated logos / images", "images"),
                     ("Repeated form objects", "forms"), ("Diagonal text objects", "rotated"),
                     ("Repeated text (footer / URL)", "repeated")):
            ttk.Checkbutton(box, text=t, variable=self.v[k]).pack(anchor="w")
        ttk.Label(t1, text="Erase boxes x0,y0,x1,y1 (0-1 from top-left; ; separates)").pack(anchor="w", pady=(6, 0))
        ttk.Entry(t1, textvariable=self.erase, width=40).pack(anchor="w")
        box = ttk.LabelFrame(t1, text="Stage 2 - pixel cleaning (GPU)", padding=6); box.pack(fill="x", pady=6)
        ttk.Label(box, text="Rasterise pages:").pack(anchor="w")
        ttk.Combobox(box, textvariable=self.v["raster"], values=["auto", "always", "never"], width=10,
                     state="readonly").pack(anchor="w")
        ttk.Checkbutton(box, text="Multi-page watermark model", variable=self.v["multipage"]).pack(anchor="w")
        ttk.Checkbutton(box, text="Single-page tone cleaning (fades light greys)", variable=self.v["levels"]).pack(anchor="w")
        ttk.Checkbutton(box, text="Repeated dark marks (footers on scans)", variable=self.v["repdark"]).pack(anchor="w")
        for t, k, lo, hi in (("Ink protect (keep darker than)", "ink", 0.2, 0.9),
                             ("Watermark cut (whiten lighter than)", "cut", 0.5, 0.99),
                             ("Multi-page floor", "floor", 0.1, 0.9)):
            ttk.Label(box, text=t).pack(anchor="w")
            ttk.Scale(box, from_=lo, to=hi, variable=self.v[k], length=260).pack(anchor="w")

        ttk.Checkbutton(t2, text="Vectorise B/W pages (true vector, tiny file, sharp at any zoom)",
                        variable=self.v["vector"]).pack(anchor="w", pady=(0, 4))
        ttk.Label(t2, text="Pages with colour when vectorising:").pack(anchor="w")
        ttk.Combobox(t2, textvariable=self.v["colorpages"], values=["keep", "grey"], width=8,
                     state="readonly").pack(anchor="w", pady=(0, 8))
        ttk.Label(t2, text="Render DPI for vector pages (scans use their own DPI):").pack(anchor="w")
        ttk.Combobox(t2, textvariable=self.v["dpi"], values=["150", "200", "300", "400", "600"], width=8).pack(anchor="w")
        ttk.Label(t2, text="Output DPI (0 = keep; type e.g. 1200 or 1600):").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(t2, textvariable=self.v["outdpi"], values=["0", "300", "600", "1200", "1600"], width=8).pack(anchor="w")
        ttk.Label(t2, text="Upscale method (used when Output DPI > scan DPI):").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(t2, textvariable=self.v["upscale"], values=["bicubic", "smooth", "ai"], width=10,
                     state="readonly").pack(anchor="w")
        ttk.Label(t2, text="AI model (method = ai):").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(t2, textvariable=self.v["aimodel"], values=list(MODELS), width=24, state="readonly").pack(anchor="w")
        ttk.Checkbutton(t2, text="Greyscale output (3x smaller)", variable=self.v["gray"]).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(t2, text="Lossless PNG (sharper, larger)", variable=self.v["png"]).pack(anchor="w")
        ttk.Checkbutton(t2, text="Safe mode (memory guards - keep on)", variable=self.v["safe"]).pack(anchor="w")
        ttk.Label(t2, foreground="#a60", wraplength=330,
                  text="AI can alter digits and thin symbols - compare with the original before trusting it. "
                       "'smooth' never invents detail. 1200+ DPI makes very large files and is slow.").pack(anchor="w", pady=8)

        row = ttk.Frame(left); row.pack(fill="x", pady=4)
        ttk.Label(row, text="Preview page").pack(side="left")
        ttk.Spinbox(row, from_=1, to=9999, textvariable=self.v["page"], width=5).pack(side="left", padx=4)
        self.btn_prev = ttk.Button(row, text="Preview", command=self.preview); self.btn_prev.pack(side="left")
        ttk.Button(row, text="Inspect footer", command=self.inspect).pack(side="left", padx=(6, 0))
        row = ttk.Frame(left); row.pack(fill="x", pady=8)
        self.btn_run = ttk.Button(row, text="Process whole PDF", command=self.run)
        self.btn_run.pack(side="left", fill="x", expand=True)
        self.btn_cancel = ttk.Button(row, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=(6, 0))
        self.bar = ttk.Progressbar(left, maximum=1.0); self.bar.pack(fill="x")
        self.status = ttk.Label(left, text="Ready", wraplength=340); self.status.pack(anchor="w", pady=4)
        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)
        self.lb, self.la = ttk.Label(right, text="before"), ttk.Label(right, text="after")
        self.lb.pack(side="left", expand=True); self.la.pack(side="left", expand=True)

    def apply_preset(self):
        o = PRESETS[self.v["preset"].get()]
        for k, val in (("structural", o.structural), ("repeated", o.remove_repeated_text),
                       ("multipage", o.multipage), ("levels", o.levels), ("repdark", o.repeat_dark),
                       ("png", o.use_png), ("gray", o.gray), ("safe", o.safe), ("raster", o.raster),
                       ("upscale", o.upscale), ("aimodel", o.ai_model), ("vector", o.vectorize)):
            self.v[k].set(val)
        self.v["dpi"].set(str(o.dpi)); self.v["outdpi"].set(str(o.out_dpi))

    def _pick(self, var, save):
        p = (filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")]) if save
             else filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")]))
        if p:
            var.set(p)
            if not save and not self.dst.get():
                self.dst.set(p[:-4] + "_clean.pdf")

    def _int(self, key, default, lo, hi):
        try:
            return max(lo, min(hi, int(float(self.v[key].get()))))
        except (ValueError, tk.TclError):
            return default

    def opts(self):
        g = {k: v.get() for k, v in self.v.items()}
        return Options(structural=g["structural"], remove_images=g["images"], remove_forms=g["forms"],
                       remove_rotated_text=g["rotated"], remove_repeated_text=g["repeated"],
                       erase_boxes=self.erase.get(), raster=g["raster"],
                       dpi=self._int("dpi", 200, 72, 600), out_dpi=self._int("outdpi", 0, 0, 2400),
                       upscale=g["upscale"], ai_model=g["aimodel"], gray=g["gray"], safe=g["safe"], vectorize=g["vector"], color_pages=g["colorpages"],
                       multipage=g["multipage"], levels=g["levels"], repeat_dark=g["repdark"], use_png=g["png"],
                       jpeg_quality=PRESETS[g["preset"]].jpeg_quality,
                       ink_t=g["ink"], wm_cut=g["cut"], floor=g["floor"])

    def _busy(self, on):
        for b, s in ((self.btn_run, not on), (self.btn_prev, not on)):
            b.config(state="normal" if s else "disabled")
        self.btn_cancel.config(state="normal" if on else "disabled")

    def _bg(self, fn):
        def wrapper():
            try:
                fn()
            except Exception as e:  # surface errors in the UI thread
                msg = str(e)
                self.after(0, lambda: (self.status.config(text=f"Error: {msg}"), messagebox.showerror("Error", msg)))
            finally:
                self.after(0, lambda: self._busy(False))
        self._busy(True)
        threading.Thread(target=wrapper, daemon=True).start()

    def inspect(self):
        if not self.src.get():
            return
        from .vector import inspect_page
        n = len(fitz.open(self.src.get()))
        pno = min(max(self.v["page"].get(), 1), n) - 1
        win = tk.Toplevel(self)
        win.title("What is in the bottom 12% of this page")
        txt = tk.Text(win, width=110, height=18, wrap="word")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", inspect_page(self.src.get(), pno))

    def preview(self):
        if not self.src.get():
            return
        o = self.opts()
        def work():
            n = len(fitz.open(self.src.get()))
            pno = min(max(self.v["page"].get(), 1), n) - 1
            b, a = preview_page(self.src.get(), pno, o)
            b.thumbnail((560, 760), Image.LANCZOS); a.thumbnail((560, 760), Image.LANCZOS)
            def show():
                self._ib, self._ia = ImageTk.PhotoImage(b), ImageTk.PhotoImage(a)
                self.lb.config(image=self._ib); self.la.config(image=self._ia)
                self.status.config(text=f"Preview of page {pno + 1} (Output DPI / upscaling not applied in preview)")
            self.after(0, show)
        self.status.config(text="Rendering preview...")
        self._bg(work)

    def run(self):
        if not (self.src.get() and self.dst.get()):
            messagebox.showwarning("Missing", "Choose input and output files."); return
        ctx = mp.get_context("spawn")
        self.q, self.cancel_mp = ctx.Queue(), ctx.Event()
        self.proc = ctx.Process(target=worker, daemon=True,
                                args=(self.src.get(), self.dst.get(), self.opts(), self.q, self.cancel_mp))
        self._busy(True)
        self.bar.configure(value=0)
        self.status.config(text="Starting worker...")
        self.proc.start()
        self.after(200, self._poll)

    def cancel(self):
        if self.cancel_mp is not None:
            self.cancel_mp.set()
            self.status.config(text="Cancelling after the current page...")

    def _poll(self):
        msg = None
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] != "p":
                    break
                self.bar.configure(value=msg[1])
                self.status.config(text=msg[2])
                msg = None
        except queue.Empty:
            pass
        if msg is None and not self.proc.is_alive():
            try:
                msg = self.q.get(timeout=1.0)
            except queue.Empty:
                msg = ("err", "The worker stopped unexpectedly (probably out of memory). Try a lower Output DPI.")
        if msg is None:
            self.after(200, self._poll)
            return
        self._busy(False)
        if msg[0] == "err":
            self.status.config(text=f"Error: {msg[1]}")
            messagebox.showerror("Error", msg[1])
            return
        rep = msg[1]
        if rep.get("cancelled"):
            self.status.config(text="Cancelled - no file written.")
            return
        self.status.config(text=f"Done: {rep.get('pages')} pages, {rep.get('rasterised_pages')} cleaned.")
        notes = rep.get("warnings", []) + rep.get("failed_pages", [])
        if notes:
            messagebox.showinfo("Finished with notes", "\n".join(notes[:12]))

def run():
    App().mainloop()
