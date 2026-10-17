# Creates a desktop shortcut that starts the app without any console window.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath('Desktop')
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut("$desktop\PDF Watermark Remover.lnk")
$lnk.TargetPath = "$root\.venv\Scripts\pythonw.exe"
$lnk.Arguments = "-m wm_remover"
$lnk.WorkingDirectory = $root
$lnk.Description = "PDF Watermark Remover (GPU)"
$lnk.Save()
Write-Host "Shortcut created on your Desktop."
