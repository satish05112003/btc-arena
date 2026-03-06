# ============================================================
#  BTC Arena AI — Desktop Shortcut Creator
#  Run this once to create the desktop shortcut automatically.
#  Usage: Right-click > Run with PowerShell
# ============================================================

$batPath     = "C:\Users\satis\OneDrive\Desktop\arena-btc\launch_arena.bat"
$icoPath     = "C:\Users\satis\OneDrive\Desktop\arena-btc\btc_icon.ico"  # optional
$shortcutDst = "$env:USERPROFILE\Desktop\BTC Arena AI.lnk"
$workingDir  = "C:\Users\satis\OneDrive\Desktop\arena-btc"

Write-Host ""
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host "   BTC Arena AI — Shortcut Creator" -ForegroundColor Cyan
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host ""

# Create the shortcut
$WshShell  = New-Object -ComObject WScript.Shell
$Shortcut  = $WshShell.CreateShortcut($shortcutDst)
$Shortcut.TargetPath       = "cmd.exe"
$Shortcut.Arguments        = "/c `"$batPath`""
$Shortcut.WorkingDirectory = $workingDir
$Shortcut.Description      = "BTC Arena AI — Bitcoin Prediction Engine"
$Shortcut.WindowStyle      = 1  # 1 = Normal window

# Apply icon only if the .ico file exists
if (Test-Path $icoPath) {
    $Shortcut.IconLocation = $icoPath
    Write-Host "  [OK] Custom Bitcoin icon applied." -ForegroundColor Green
} else {
    Write-Host "  [INFO] btc_icon.ico not found — using default icon." -ForegroundColor Yellow
    Write-Host "         Download a .ico from https://icons8.com/icons/set/bitcoin" -ForegroundColor Gray
    Write-Host "         Save it as: $icoPath" -ForegroundColor Gray
}

$Shortcut.Save()

Write-Host ""
Write-Host "  [OK] Shortcut created on Desktop:" -ForegroundColor Green
Write-Host "       $shortcutDst" -ForegroundColor White
Write-Host ""
Write-Host "  Double-click 'BTC Arena AI' on your Desktop to launch!" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Press any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
