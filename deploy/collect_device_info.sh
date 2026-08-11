#!/usr/bin/env bash
ADB="${HOME}/android/platform-tools/adb"
if [[ ! -x "$ADB" ]]; then ADB="adb"; fi

if ! command -v "$ADB" >/dev/null 2>&1; then
  echo '{"error":"adb not found"}' >&2
  exit 1
fi

if ! "$ADB" get-state >/dev/null 2>&1; then
  echo '{"error":"no adb device — enable USB debugging and authorize PC"}' >&2
  exit 1
fi

python3 - <<'PY'
import json
import subprocess

import os
ADB = os.environ.get("ADB", os.path.expanduser("~/android/platform-tools/adb"))

def sh(*args):
    return subprocess.check_output([ADB, "shell", *args], text=True, stderr=subprocess.DEVNULL).strip()

def prop(key):
    try:
        return sh("getprop", key)
    except Exception:
        return ""

info = {
    "manufacturer": prop("ro.product.manufacturer"),
    "model": prop("ro.product.model"),
    "device": prop("ro.product.device"),
    "brand": prop("ro.product.brand"),
    "soc_model": prop("ro.soc.model") or prop("ro.board.platform"),
    "android_release": prop("ro.build.version.release"),
    "sdk": prop("ro.build.version.sdk"),
    "build_id": prop("ro.build.id"),
    "gpu": "",
}
try:
    info["gpu"] = sh("dumpsys", "SurfaceFlinger", "|", "grep", "GLES").split("\n")[0]
except Exception:
    pass

print(json.dumps(info, indent=2))
PY
