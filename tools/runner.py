#!/usr/bin/env python3
"""Run all Jupyter notebooks in a directory and its subdirectories."""

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def run_notebook(notebook_path: Path, timeout: int, kernel: str) -> tuple[bool, str]:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
        return True, ""
    except CellExecutionError as e:
        return False, str(e)
    except TimeoutError:
        return False, f"Timed out after {timeout}s"


def find_notebooks(root: Path, pattern: str = "*.ipynb") -> list[Path]:
    return sorted(root.rglob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Run all notebooks in a directory tree")
    parser.add_argument("directory", nargs="?", default=".", help="Root directory to search (default: current dir)")
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell timeout in seconds (default: 600)")
    parser.add_argument("--kernel", default="python3", help="Kernel name to use (default: python3)")
    parser.add_argument("--pattern", default="*.ipynb", help="Glob pattern for notebooks (default: *.ipynb)")
    parser.add_argument("--ignore", nargs="*", default=[], help="Substrings to skip in notebook paths")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after first failure")
    parser.add_argument("--inplace", action="store_true", help="Save executed output back to notebooks")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    notebooks = find_notebooks(root, args.pattern)
    if args.ignore:
        notebooks = [
            nb for nb in notebooks
            if not any(skip in str(nb) for skip in args.ignore)
        ]

    if not notebooks:
        print(f"No notebooks found in {root}")
        sys.exit(0)

    print(f"Found {len(notebooks)} notebook(s) under {root}\n")

    passed, failed = [], []
    for nb_path in notebooks:
        rel = nb_path.relative_to(root)
        print(f"Running {rel} ... ", end="", flush=True)
        start = time.monotonic()
        ok, err = run_notebook(nb_path, args.timeout, args.kernel)
        elapsed = time.monotonic() - start

        if ok:
            print(f"OK ({elapsed:.1f}s)")
            passed.append(rel)
            if args.inplace:
                with open(nb_path) as f:
                    nb = nbformat.read(f, as_version=4)
                ep = ExecutePreprocessor(timeout=args.timeout, kernel_name=args.kernel)
                ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
                with open(nb_path, "w") as f:
                    nbformat.write(nb, f)
        else:
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"  {err[:300]}")
            failed.append(rel)
            if args.stop_on_error:
                break

    print(f"\nResults: {len(passed)} passed, {len(failed)} failed out of {len(passed) + len(failed)} run")
    if failed:
        print("\nFailed notebooks:")
        for nb in failed:
            print(f"  {nb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
