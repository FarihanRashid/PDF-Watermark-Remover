# OPTIONAL: builds a standalone WMRemover.exe (folder build). Expect 3-5 GB (PyTorch + CUDA libraries) and a long build.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
pip install pyinstaller
$args = @("--noconfirm", "--windowed", "--name", "WMRemover", "--collect-all", "torch", "--collect-all", "pymupdf")
if (pip show spandrel 2>$null) { $args += @("--collect-all", "spandrel") }
pyinstaller @args run.py
Write-Host "Done. Run dist\WMRemover\WMRemover.exe  (copy the whole WMRemover folder to move it)."
