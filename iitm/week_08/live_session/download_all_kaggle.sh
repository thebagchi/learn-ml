#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DOWNLOADER="${REPO_ROOT}/tools/kaggle_downloader.py"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"

if [[ ! -f "${DOWNLOADER}" ]]; then
  echo "Error: downloader not found at ${DOWNLOADER}" >&2
  exit 1
fi

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  echo "Error: KAGGLE_USERNAME and KAGGLE_KEY must be set in environment." >&2
  exit 1
fi

URLS=(
  "https://www.kaggle.com/competitions/titanic"
  "https://www.kaggle.com/datasets/uciml/iris"
  "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data"
)

for url in "${URLS[@]}"; do
  echo "Downloading: ${url}"
  "${PYTHON_BIN}" "${DOWNLOADER}" "${url}" -o "${SCRIPT_DIR}"
done

echo "All datasets downloaded into ${SCRIPT_DIR}"