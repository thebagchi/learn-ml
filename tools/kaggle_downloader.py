#!/usr/bin/env python3
"""
Kaggle Dataset/Competition Downloader

Download files from a Kaggle dataset or competition URL using the Kaggle CLI.

USAGE:
    Dataset URL:
        python tools/kaggle_downloader.py \
            https://www.kaggle.com/datasets/uciml/iris

    Competition URL:
        python tools/kaggle_downloader.py \
            https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

    Custom output directory (auto-unzips by default):
        python tools/kaggle_downloader.py <kaggle_url> -o data/raw

    Skip extraction:
        python tools/kaggle_downloader.py <kaggle_url> -o data/raw --no-unzip

PREREQUISITES:
    - Kaggle CLI installed (pip install kaggle)
    - Prefer env vars KAGGLE_USERNAME and KAGGLE_KEY
    - Or KAGGLE_API_TOKEN in env (see formats below)
    - Or Kaggle API credentials configured (~/.kaggle/kaggle.json)

KAGGLE_API_TOKEN formats:
    - JSON: {"username":"<user>","key":"<api_key>"}
    - <username>:<api_key>
    - <api_key> with KAGGLE_USERNAME (or KAGGLE_USER) set
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def parse_kaggle_target(value: str) -> tuple[str, str]:
    """
    Return (kind, identifier) where kind is one of: dataset, competition.

    Accepted URL formats:
    - https://www.kaggle.com/datasets/<owner>/<dataset>
    - https://www.kaggle.com/competitions/<competition>
    """
    parsed = urlparse(value)
    host = (parsed.netloc or "").lower()

    if "kaggle.com" not in host:
        raise ValueError("Expected a Kaggle URL (kaggle.com).")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "datasets":
        owner, dataset = parts[1], parts[2]
        return "dataset", f"{owner}/{dataset}"

    if len(parts) >= 2 and parts[0] == "competitions":
        competition = parts[1]
        return "competition", competition

    raise ValueError(
        "Unsupported Kaggle URL. Use a dataset or competition URL, for example: "
        "https://www.kaggle.com/datasets/uciml/iris"
    )


def unzip_archives(output_dir: Path) -> None:
    """Unzip all .zip files in output_dir into the same directory."""
    zip_files = sorted(output_dir.glob("*.zip"))
    if not zip_files:
        print("No ZIP files found to extract.")
        return

    for zf in zip_files:
        print(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as archive:
            archive.extractall(output_dir)


def dataset_subdir_name(kind: str, identifier: str) -> str:
    """Create a filesystem-safe subdirectory name from Kaggle identifier."""
    if kind == "dataset":
        # identifier is owner/dataset; keep the dataset slug for directory naming.
        name = identifier.split("/", 1)[1]
    else:
        name = identifier

    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "kaggle_data"


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> int:
    try:
        completed = subprocess.run(cmd, check=False, env=env)
        return completed.returncode
    except FileNotFoundError:
        print(
            'Error: "kaggle" CLI not found. Install it with "pip install kaggle".',
            file=sys.stderr,
        )
        return 1


def build_kaggle_env() -> tuple[dict[str, str], str | None]:
    """
    Build environment for Kaggle CLI.

    If KAGGLE_USERNAME and KAGGLE_KEY are present, write a temporary kaggle.json
    and point KAGGLE_CONFIG_DIR to it.

    Else, if KAGGLE_API_TOKEN is present, try to derive credentials and write
    a temporary kaggle.json.

    Supported KAGGLE_API_TOKEN formats:
    - JSON: {"username":"<user>","key":"<api_key>"}
    - username:key
    - key only (requires KAGGLE_USERNAME in env)
    """
    env = os.environ.copy()
    username = env.get("KAGGLE_USERNAME", "").strip()
    key = env.get("KAGGLE_KEY", "").strip()

    if username and key:
        temp_dir = tempfile.mkdtemp(prefix="kaggle_cfg_")
        config_file = Path(temp_dir) / "kaggle.json"
        config_file.write_text(
            json.dumps({"username": username, "key": key}), encoding="utf-8"
        )
        os.chmod(config_file, 0o600)
        env["KAGGLE_CONFIG_DIR"] = temp_dir
        return env, temp_dir

    token = env.get("KAGGLE_API_TOKEN", "").strip()

    if not token:
        return env, None

    username = ""
    key = ""

    try:
        payload = json.loads(token)
        if isinstance(payload, dict):
            username = str(payload.get("username", "")).strip()
            key = str(payload.get("key", "")).strip()
    except json.JSONDecodeError:
        if ":" in token:
            username, key = [part.strip() for part in token.split(":", 1)]
        else:
            username = env.get("KAGGLE_USERNAME", "").strip() or env.get(
                "KAGGLE_USER", ""
            ).strip()
            key = token

    if not username or not key:
        raise ValueError(
            "KAGGLE_API_TOKEN is set but invalid. Use JSON with username/key, "
            "username:key, or set KAGGLE_USERNAME/KAGGLE_USER with token as key."
        )

    temp_dir = tempfile.mkdtemp(prefix="kaggle_cfg_")
    config_file = Path(temp_dir) / "kaggle.json"
    config_file.write_text(
        json.dumps({"username": username, "key": key}), encoding="utf-8"
    )
    os.chmod(config_file, 0o600)
    env["KAGGLE_CONFIG_DIR"] = temp_dir

    return env, temp_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download data from Kaggle URL.")
    parser.add_argument("url", help="Kaggle dataset or competition URL")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="downloads/kaggle",
        help="Directory to store downloaded files (default: downloads/kaggle)",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not extract downloaded ZIP files",
    )

    args = parser.parse_args()

    output_root = Path(os.path.expanduser(args.output_dir)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        kind, identifier = parse_kaggle_target(args.url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if kind == "dataset":
        target_dir = output_root / dataset_subdir_name(kind, identifier)
        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", identifier, "-p", str(target_dir)]
    else:
        target_dir = output_root / dataset_subdir_name(kind, identifier)
        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "competitions", "download", "-c", identifier, "-p", str(target_dir)]

    try:
        env, temp_config_dir = build_kaggle_env()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Detected {kind}: {identifier}")
    print("Running:", " ".join(cmd))
    exit_code = run_command(cmd, env=env)

    if temp_config_dir:
        shutil.rmtree(temp_config_dir, ignore_errors=True)

    if exit_code != 0:
        print(
            "Download failed. Provide KAGGLE_API_TOKEN or configure Kaggle "
            "credentials at ~/.kaggle/kaggle.json (or ~/.config/kaggle/kaggle.json), "
            "and accept competition rules when required.",
            file=sys.stderr,
        )
        return exit_code

    print(f"Download complete: {target_dir}")

    if not args.no_unzip:
        unzip_archives(target_dir)
        print("Extraction complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
