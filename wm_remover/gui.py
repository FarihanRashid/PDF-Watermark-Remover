import multiprocessing as mp
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pymupdf as fitz
from PIL import Image, ImageTk

from .pipeline import PRESETS, Options, collect_pdfs, preview_page, worker_batch
from .raster import MODELS, gpu_info

try:                                    # optional: native drag & drop (pip install tkinterdnd2)
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _Base, HAS_DND = TkinterDnD.Tk, True
except Exception:
    _Base, HAS_DND = tk.Tk, False


class App(_Base):
    def __init__(self):
        super().__init__()
        self.title("PDF Watermark Remover (GPU)")
        self.geometry("1200x900")
        self.files, self.last_out = [], None
        self.proc, self.q, self.cancel_mp = None, None, None
        B = lambda v: tk.BooleanVar(value=v)
        S = tk.StringVar
        self.v = {
            "structural": B(True), "images": B(True), "forms": B(True), "rotated": B(True), "repeated": B(True),
            "multipage": B(True), "levels": B(False), "repdark": B(False), "png": B(True), "gray": B(False),
            "safe": B(True), "vector": B(False), "colorpages": S(value="keep"), "preset": S(value="balanced"),
            "raster": S(value="auto"), "upscale": S(value="smooth"), "aimodel": S(value=list(MODELS)[0]),
            "dpi": S(value="200"), "outdpi": S(value="0"), "erase": S(value=""), "outname": S(value="cleaned"),
            "ink": tk.DoubleVar(value=0.60), "cut": tk.DoubleVar(value=0.86), "floor": tk.DoubleVar(value=0.45),
            "page": tk.IntVar(value=1),
        }
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        if HAS_DND:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", lambda e: self.add_paths(self.tk.splitlist(e.data)))

    def _close(self):
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
        self.destroy()

    # ------------------------------------------------------------------ layout
    def _build(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        ttk.Label(left, text=gpu_info(), foreground="#0a7").pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="PDFs to process" + ("  (drop files or folders anywhere in this window)" if HAS_DND else "")
                  ).pack(anchor="w")
        row = ttk.Frame(left); row.pack(fill="x")
        self.lst = tk.Listbox(row, height=6, width=52, selectmode="extended", activestyle="none")
        sb = ttk.Scrollbar(row, command=self.lst.yview)
        self.lst.config(yscrollcommand=sb.set)
        self.lst.pack(side="left", fill="x", expand=True); sb.pack(side="left", fill="y")
        row = ttk.Frame(left); row.pack(fill="x", pady=3)
        for t, c in (("Add files...", self.pick_files), ("Add folder...", self.pick_folder),
                     ("Remove", self.remove_selected), ("Clear", self.clear_files)):
            ttk.Button(row, text=t, command=c).pack(side="left", padx=(0, 4))
        if not HAS_DND:
            ttk.Label(left, foreground="#a60", text="Drag & drop: pip install tkinterdnd2, then restart.").pack(anchor="w")
        row = ttk.Frame(left); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Save in subfolder").pack(side="left")
        ttk.Entry(row, textvariable=self.v["outname"], width=14).pack(side="left", padx=4)
        ttk.Label(row, text="next to each PDF", foreground="#666").pack(side="left")

        row = ttk.Frame(left); row.pack(fill="x", pady=(6, 0))
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
        ttk.Entry(t1, textvariable=self.v["erase"], width=40).pack(anchor="w")
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
        ttk.Label(t2, text="Output DPI (0 = keep; capped at 3x the scan DPI in safe mode):").pack(anchor="w", pady=(8, 0))
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
                       "'smooth' never invents detail.").pack(anchor="w", pady=8)

        row = ttk.Frame(left); row.pack(fill="x", pady=4)
        ttk.Label(row, text="Preview page").pack(side="left")
        ttk.Spinbox(row, from_=1, to=9999, textvariable=self.v["page"], width=5).pack(side="left", padx=4)
        self.btn_prev = ttk.Button(row, text="Preview", command=self.preview); self.btn_prev.pack(side="left")
        ttk.Button(row, text="Inspect footer", command=self.inspect).pack(side="left", padx=(6, 0))
        row = ttk.Frame(left); row.pack(fill="x", pady=8)
        self.btn_run = ttk.Button(row, text="Process all", command=self.run)
        self.btn_run.pack(side="left", fill="x", expand=True)
        self.btn_cancel = ttk.Button(row, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=(6, 0))
        self.btn_open = ttk.Button(row, text="Open output", command=self.open_output, state="disabled")
        self.btn_open.pack(side="left", padx=(6, 0))
        self.bar = ttk.Progressbar(left, maximum=1.0); self.bar.pack(fill="x")
        self.status = ttk.Label(left, text="Add PDFs to begin", wraplength=380); self.status.pack(anchor="w", pady=4)
        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)
        self.lb, self.la = ttk.Label(right, text="before"), ttk.Label(right, text="after")
        self.lb.pack(side="left", expand=True); self.la.pack(side="left", expand=True)

    # ------------------------------------------------------------------ file list
    def add_paths(self, paths):
        found = collect_pdfs(paths)
        new = [f for f in found if f not in self.files]
        for f in new:
            self.files.append(f)
            self.lst.insert("end", f)
        self.status.config(text=f"Added {len(new)} PDF(s); {len(self.files)} queued." if found
                           else "No PDF files found in what you added.")

    def pick_files(self):
        self.add_paths(filedialog.askopenfilenames(filetypes=[("PDF", "*.pdf")]))

    def pick_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.add_paths([d])

    def remove_selected(self):
        for i in reversed(self.lst.curselection()):
            self.lst.delete(i)
            del self.files[i]

    def clear_files(self):
        self.lst.delete(0, "end")
        self.files.clear()

    def _selected(self):
        sel = self.lst.curselection()
        return self.files[sel[0]] if sel else (self.files[0] if self.files else None)

    def open_output(self):
        if self.last_out and os.path.isdir(self.last_out):
            os.startfile(self.last_out)

    # ------------------------------------------------------------------ options
    def apply_preset(self):
        o = PRESETS[self.v["preset"].get()]
        for k, val in (("structural", o.structural), ("repeated", o.remove_repeated_text),
                       ("multipage", o.multipage), ("levels", o.levels), ("repdark", o.repeat_dark),
                       ("png", o.use_png), ("gray", o.gray), ("safe", o.safe), ("raster", o.raster),
                       ("upscale", o.upscale), ("aimodel", o.ai_model), ("vector", o.vectorize)):
            self.v[k].set(val)
        self.v["dpi"].set(str(o.dpi)); self.v["outdpi"].set(str(o.out_dpi))

    def _int(self, key, default, lo, hi):
        try:
            return max(lo, min(hi, int(float(self.v[key].get()))))
        except (ValueError, tk.TclError):
            return default

    def opts(self):
        g = {k: v.get() for k, v in self.v.items()}
        return Options(structural=g["structural"], remove_images=g["images"], remove_forms=g["forms"],
                       remove_rotated_text=g["rotated"], remove_repeated_text=g["repeated"],
                       erase_boxes=g["erase"], raster=g["raster"],
                       dpi=self._int("dpi", 200, 72, 600), out_dpi=self._int("outdpi", 0, 0, 2400),
                       upscale=g["upscale"], ai_model=g["aimodel"], gray=g["gray"], safe=g["safe"],
                       vectorize=g["vector"], color_pages=g["colorpages"],
                       multipage=g["multipage"], levels=g["levels"], repeat_dark=g["repdark"], use_png=g["png"],
                       jpeg_quality=PRESETS[g["preset"]].jpeg_quality,
                       ink_t=g["ink"], wm_cut=g["cut"], floor=g["floor"])

    # ------------------------------------------------------------------ preview / inspect (single file, thread)
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
        src = self._selected()
        if not src:
            return
        from .vector import inspect_page
        n = len(fitz.open(src))
        pno = min(max(self.v["page"].get(), 1), n) - 1
        win = tk.Toplevel(self)
        win.title("What is in the bottom 12% of this page")
        txt = tk.Text(win, width=110, height=18, wrap="word")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", inspect_page(src, pno))

    def preview(self):
        src = self._selected()
        if not src:
            self.status.config(text="Add a PDF first.")
            return
        o = self.opts()
        def work():
            n = len(fitz.open(src))
            pno = min(max(self.v["page"].get(), 1), n) - 1
            b, a = preview_page(src, pno, o)
            b.thumbnail((560, 760), Image.LANCZOS); a.thumbnail((560, 760), Image.LANCZOS)
            def show():
                self._ib, self._ia = ImageTk.PhotoImage(b), ImageTk.PhotoImage(a)
                self.lb.config(image=self._ib); self.la.config(image=self._ia)
                self.status.config(text=f"Preview: {os.path.basename(src)}, page {pno + 1} "
                                        "(Output DPI / upscaling not applied in preview)")
            self.after(0, show)
        self.status.config(text="Rendering preview...")
        self._bg(work)

    # ------------------------------------------------------------------ batch run (separate process)
    def run(self):
        if not self.files:
            messagebox.showwarning("Nothing to do", "Add PDF files or a folder first."); return
        name = self.v["outname"].get().strip().strip("\\/") or "cleaned"
        ctx = mp.get_context("spawn")
        self.q, self.cancel_mp = ctx.Queue(), ctx.Event()
        self.proc = ctx.Process(target=worker_batch, daemon=True,
                                args=(list(self.files), self.opts(), name, self.q, self.cancel_mp))
        self._busy(True)
        self.btn_open.config(state="disabled")
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
        self._show_summary(msg[1])

    def _show_summary(self, summ):
        res = summ["results"]
        ok = [r for r in res if r["status"] == "ok"]
        bad = [r for r in res if r["status"] == "error"]
        skipped = [r for r in res if r["status"] == "skipped"]
        if not skipped:
            self.bar.configure(value=1.0)
        self.status.config(text=f"Done: {len(ok)} of {summ['files']} file(s) OK"
                                + (f", {len(bad)} failed" if bad else "")
                                + (f", {len(skipped)} skipped (cancelled)" if skipped else "")
                                + f". Saved in the '{summ['out_name']}' folder next to each PDF.")
        if ok:
            self.last_out = os.path.dirname(ok[0]["output"])
            self.btn_open.config(state="normal")
        notes = [f"FAILED {r['file']}: {r['error']}" for r in bad]
        for r in ok:
            notes += [f"{r['file']}: {w}" for w in r["warnings"]] + [f"{r['file']}: {w}" for w in r["failed_pages"]]
        if notes:
            messagebox.showinfo("Finished with notes", "\n".join(notes[:15]) + ("\n..." if len(notes) > 15 else ""))


def run():
    App().mainloop()
