#!/usr/bin/env bash
# Cross-compile sr_bench for Android arm64 (NCNN + Vulkan).
#
# Prerequisites:
#   - Android NDK (r26+), e.g. $HOME/Android/Sdk/ndk/26.1.10909125
#   - NCNN built for Android (this script can build it)
#
# Usage:
#   export ANDROID_NDK=$HOME/Android/Sdk/ndk/26.1.10909125
#   ./deploy/build_android_bench.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NCNN_SRC="${NCNN_SRC:-$HOME/ncnn}"
NDK="${ANDROID_NDK:-${ANDROID_NDK_HOME:-$HOME/android/ndk/android-ndk-r26d}}"
BUILD_DIR="$ROOT/deploy/android/sr_bench/build"
INSTALL_DIR="$NCNN_SRC/build-android-aarch64/install"
BENCH_BIN="$BUILD_DIR/sr_bench"

if [[ -z "$NDK" || ! -d "$NDK" ]]; then
  echo "ERROR: set ANDROID_NDK to your NDK path (see deploy/DEPLOY.md)" >&2
  exit 1
fi

if [[ ! -d "$NCNN_SRC/.git" ]]; then
  echo "Cloning NCNN into $NCNN_SRC ..."
  git clone --depth=1 https://github.com/Tencent/ncnn.git "$NCNN_SRC"
fi

if [[ ! -f "$INSTALL_DIR/lib/libncnn.a" ]]; then
  echo "Building NCNN for Android arm64-v8a (CPU; Vulkan optional)..."
  mkdir -p "$NCNN_SRC/build-android-aarch64"
  cmake -S "$NCNN_SRC" -B "$NCNN_SRC/build-android-aarch64" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DNCNN_VULKAN=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
  cmake --build "$NCNN_SRC/build-android-aarch64" -j"$(nproc)"
  cmake --install "$NCNN_SRC/build-android-aarch64"
fi

echo "Building sr_bench..."
cmake -S "$ROOT/deploy/android/sr_bench" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DNCNN_DIR="$INSTALL_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "Built: $BENCH_BIN"
file "$BENCH_BIN" || true
