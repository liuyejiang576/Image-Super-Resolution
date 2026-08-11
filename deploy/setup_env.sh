#!/usr/bin/env bash
# Source before deployment commands:
#   source deploy/setup_env.sh
export ANDROID_HOME="${ANDROID_HOME:-$HOME/android}"
export ANDROID_NDK="${ANDROID_NDK:-$HOME/android/ndk/android-ndk-r26d}"
export PATH="$HOME/android/platform-tools:$HOME/ncnn/build-host/tools/onnx:$HOME/ncnn/build-host/tools:$PATH"
export ONNX2NCNN="${ONNX2NCNN:-$HOME/ncnn/build-host/tools/onnx/onnx2ncnn}"
export NCNNOPT="${NCNNOPT:-$HOME/ncnn/build-host/tools/ncnnoptimize}"
