#!/usr/bin/env python3
"""
Stanford CS229 Materials Downloader

Download course materials from Stanford's CS229 machine learning course.
https://see.stanford.edu/Course/CS229

Delegates to downloader.go (via go run) for fetching and downloading.

USAGE:
    Basic usage (downloads to 'cs229_materials/' directory):
        python download_stanford_cs229.py

    Specify custom output directory:
        python download_stanford_cs229.py /path/to/custom/directory

    Non-interactive (skip confirmation prompt):
        python download_stanford_cs229.py -y

    Show this help:
        python download_stanford_cs229.py --help

PREREQUISITES:
    Go 1.18+ must be installed and available on PATH.
"""

import os
import subprocess
import sys

CS229_URL = 'https://see.stanford.edu/Course/CS229'
DEFAULT_OUTPUT_DIR = 'cs229_materials'
DOWNLOADER_GO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloader.go')


def show_help():
    print(__doc__)


def main():
    args = sys.argv[1:]

    if args and args[0] in ('-h', '--help', 'help'):
        show_help()
        sys.exit(0)

    # Separate known flags from positional output-dir argument
    yes_flag = '-y' in args
    positional = [a for a in args if a != '-y']

    output_dir = positional[0] if positional else DEFAULT_OUTPUT_DIR
    if output_dir.startswith('~'):
        output_dir = os.path.expanduser(output_dir)

    cmd = [
        'go', 'run', DOWNLOADER_GO,
        '-url', CS229_URL,
        '-d', output_dir,
        '-keywords', 'lecture,session',
    ]
    if yes_flag:
        cmd.append('-y')

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print('Error: "go" not found. Please install Go 1.18+ and ensure it is on your PATH.',
              file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == '__main__':
    main()
