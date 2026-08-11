#!/usr/bin/env bash
# Build sr_bench for Linux host (smoke-test NCNN models before phone deploy).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NCNN_SRC="${NCNN_SRC:-$HOME/ncnn}"
BUILD_DIR="$ROOT/deploy/android/sr_bench/build-host"
INSTALL_DIR="$NCNN_SRC/build-host/install"

if [[ ! -f "$NCNN_SRC/build-host/src/libncnn.a" ]]; then
  echo "ERROR: host NCNN not built at $NCNN_SRC/build-host" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR/lib" "$INSTALL_DIR/include"
if [[ ! -f "$INSTALL_DIR/lib/libncnn.a" ]]; then
  cp "$NCNN_SRC/build-host/src/libncnn.a" "$INSTALL_DIR/lib/"
  cp -r "$NCNN_SRC/src" "$NCNN_SRC/build-host/install/include/ncnn_src" 2>/dev/null || true
  rsync -a --include='*.h' --include='*/' --exclude='*' "$NCNN_SRC/src/" "$INSTALL_DIR/include/" 2>/dev/null || \
    find "$NCNN_SRC/src" -name '*.h' -exec cp --parents {} "$INSTALL_DIR/include/" \; 2>/dev/null || true
fi

# Use ncnn source headers directly
cmake -S "$ROOT/deploy/android/sr_bench" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_DIR="$NCNN_SRC/build-host"
# Patch: link against build tree
cmake -S "$ROOT/deploy/android/sr_bench" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_DIR="$NCNN_SRC/build-host" 2>/dev/null || true

cat > "$BUILD_DIR/ncnn_host.cmake" <<'EOF'
# Host override injected by build_host_bench.sh
EOF

cmake -S "$ROOT/deploy/android/sr_bench" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_DIR="$NCNN_SRC/build-host"

# Fix CMakeLists for host - simpler to compile inline
g++ -O3 -std=c++14 -I"$NCNN_SRC/src" -I"$NCNN_SRC/build-host/src" \
  "$ROOT/deploy/android/sr_bench/main.cpp" \
  -L"$NCNN_SRC/build-host/src" -lncnn -fopenmp -pthread \
  -o "$BUILD_DIR/sr_bench_host"

echo "Built host smoke binary: $BUILD_DIR/sr_bench_host"
