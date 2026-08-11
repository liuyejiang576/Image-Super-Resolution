#!/usr/bin/env bash
# Check adb connectivity and print fix steps for WSL2 USB.
set -euo pipefail
ADB="${HOME}/android/platform-tools/adb"
if [[ ! -x "$ADB" ]]; then
  ADB="adb"
fi

echo "=== adb devices ==="
"$ADB" devices -l || true
echo

if "$ADB" get-state >/dev/null 2>&1; then
  echo "OK: device ready for bench_mobile.py"
  exit 0
fi

echo "No device visible in WSL."
echo
echo "Your phone is likely connected to Windows, not WSL. Do ONE of:"
echo
echo "A) Windows 11 — attach USB to WSL (Settings → USB → attach device to WSL)"
echo
echo "B) usbipd-win (PowerShell as Admin on Windows):"
echo "   winget install dorssel.usbipd"
echo "   usbipd list"
echo "   usbipd bind --busid <BUSID>"
echo "   usbipd attach --wsl --busid <BUSID>"
echo "   # then re-run: deploy/check_usb.sh"
echo
echo "C) Wireless adb (phone and PC same Wi‑Fi):"
echo "   On phone: Developer options → Wireless debugging → Pair device"
echo "   adb pair <ip>:<port>   # enter pairing code"
echo "   adb connect <ip>:<port>"
echo
echo "Also on phone: USB debugging ON, authorize this PC when prompted."
exit 1
