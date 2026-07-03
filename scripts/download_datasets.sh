#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"
BENCH_DIR="${PROJECT_ROOT}/data/benchmarks"
DIV2K_DIR="${PROJECT_ROOT}/data/div2k"

mkdir -p "${RAW_DIR}" "${BENCH_DIR}" "${DIV2K_DIR}"

declare -A EXPECTED_BYTES=(
  ["Set5.zip"]=3004654
  ["Set14.zip"]=17226584
  ["BSD100.zip"]=93680873
  ["Urban100.zip"]=189239475
  ["DIV2K_train_HR.zip"]=3530603713
  ["DIV2K_valid_HR.zip"]=448993893
)

is_complete() {
  local file="$1"
  local name expected local_size

  name="$(basename "${file}")"
  expected="${EXPECTED_BYTES[${name}]:-}"
  [[ -n "${expected}" ]] || return 1
  [[ -f "${file}" ]] || return 1
  local_size="$(stat -c%s "${file}")"
  [[ "${local_size}" == "${expected}" ]]
}

download_file() {
  local url="$1"
  local out="$2"
  local use_proxy="${3:-1}"
  local name expected local_size curl_args=()

  name="$(basename "${out}")"
  expected="${EXPECTED_BYTES[${name}]:-}"

  if is_complete "${out}"; then
    echo "Skip ${out} (already complete: ${expected} bytes)"
    return 0
  fi

  if [[ "${use_proxy}" == "0" ]]; then
    curl_args+=(--noproxy '*')
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  fi

  if [[ -f "${out}" ]]; then
    local_size="$(stat -c%s "${out}")"
    if [[ -n "${expected}" ]]; then
      echo "Resuming ${out} (${local_size}/${expected} bytes) ..."
    else
      echo "Resuming ${out} (${local_size} bytes) ..."
    fi
  else
    echo "Downloading ${out} ..."
  fi

  curl "${curl_args[@]}" -L --fail --retry 5 --retry-delay 5 -C - \
    --connect-timeout 30 -o "${out}" "${url}"
}

extract_zip() {
  local file="$1"
  local out_dir="$2"
  echo "Extracting ${file} ..."
  unzip -o "${file}" -d "${out_dir}" >/dev/null
}

if [[ -d "${BENCH_DIR}/Set5" && -d "${BENCH_DIR}/Set14" && -d "${BENCH_DIR}/BSD100" && -d "${BENCH_DIR}/Urban100" ]]; then
  echo "Benchmark folders already present; skipping benchmark downloads."
else
  # Benchmark test sets from Figshare article:
  # https://doi.org/10.6084/m9.figshare.21586188.v1
  download_file "https://ndownloader.figshare.com/files/38256852" "${RAW_DIR}/Set5.zip" 1
  download_file "https://ndownloader.figshare.com/files/38256855" "${RAW_DIR}/Set14.zip" 1
  download_file "https://ndownloader.figshare.com/files/38256840" "${RAW_DIR}/BSD100.zip" 1
  download_file "https://ndownloader.figshare.com/files/38256858" "${RAW_DIR}/Urban100.zip" 1

  extract_zip "${RAW_DIR}/Set5.zip" "${BENCH_DIR}"
  extract_zip "${RAW_DIR}/Set14.zip" "${BENCH_DIR}"
  extract_zip "${RAW_DIR}/BSD100.zip" "${BENCH_DIR}"
  extract_zip "${RAW_DIR}/Urban100.zip" "${BENCH_DIR}"
fi

# DIV2K HR images from official ETH source (direct connection is faster here).
download_file "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" "${RAW_DIR}/DIV2K_train_HR.zip" 0
download_file "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" "${RAW_DIR}/DIV2K_valid_HR.zip" 0

extract_zip "${RAW_DIR}/DIV2K_train_HR.zip" "${DIV2K_DIR}"
extract_zip "${RAW_DIR}/DIV2K_valid_HR.zip" "${DIV2K_DIR}"

echo "All datasets downloaded and extracted."
