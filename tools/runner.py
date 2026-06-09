#!/usr/bin/env python3
"""Run all Jupyter notebooks in a directory and its subdirectories."""

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)


def format_error(e: CellExecutionError, verbose: bool) -> str:
    """Format a CellExecutionError with full traceback details."""
    lines = []
    lines.append(f"  Error type : {e.ename}")
    lines.append(f"  Error value: {e.evalue}")

    # Parse cell source and traceback from e.traceback (the full error string)
    raw = e.traceback or ""
    sep = "------------------"
    parts = raw.split(sep)
    # parts[0] = "An error occurred...\n"
    # parts[1] = "\n<cell source>\n"
    # parts[2] = "\n\n<traceback>"

    if len(parts) >= 3:
        cell_src = parts[1].strip()
        tb_raw = strip_ansi(sep.join(parts[2:])).strip()
    else:
        cell_src = ""
        tb_raw = strip_ansi(raw).strip()

    # Cell source
    src_lines = cell_src.splitlines() if cell_src else []
    if src_lines:
        lines.append("  Failing cell source:")
        if verbose or len(src_lines) <= 15:
            shown = src_lines
        else:
            shown = src_lines[:15]
            lines_after = len(src_lines) - 15
        for ln in shown:
            lines.append(f"    {ln}")
        if not verbose and len(src_lines) > 15:
            lines.append(f"    ... ({lines_after} more lines — run with --verbose to see all)")

    # Full traceback
    if tb_raw:
        lines.append("  Traceback:")
        for tl in tb_raw.splitlines():
            lines.append(f"    {tl}")

    return "\n".join(lines)


def run_notebook(notebook_path: Path, timeout: int, kernel: str) -> tuple[bool, str, object, object]:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
        # Stamp kernelspec so VS Code recognises the kernel on open
        nb.metadata.setdefault("kernelspec", {}).update({
            "display_name": f"Python 3 ({kernel})",
            "language": "python",
            "name": kernel,
        })
        nb.metadata.setdefault("language_info", {}).setdefault("name", "python")
        return True, "", None, nb
    except CellExecutionError as e:
        return False, str(e), e, None
    except TimeoutError:
        return False, f"Timed out after {timeout}s", None, None


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
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full cell source and traceback on error")
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
        ok, err, exc, nb = run_notebook(nb_path, args.timeout, args.kernel)
        elapsed = time.monotonic() - start

        if ok:
            print(f"OK ({elapsed:.1f}s)")
            passed.append(rel)
            if args.inplace:
                with open(nb_path, "w") as f:
                    nbformat.write(nb, f)
        else:
            print(f"FAILED ({elapsed:.1f}s)")
            if exc is not None:
                print(format_error(exc, args.verbose))
            else:
                print(f"  {err}")
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
