#!/usr/bin/env python3
"""
Utility script to manage Jupyter kernels - display info, select, and restart/kill kernels.
Usage: python restart_kernel.py [notebook_path]
"""

import subprocess
import signal
import sys
import os
import json
from pathlib import Path
from importlib import import_module

def get_system_info():
    """Get system and environment information."""
    info = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "venv_active": hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
        "venv_path": os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment'),
    }
    return info

def get_package_versions():
    """Get versions of key packages."""
    packages = ['jupyter', 'ipython', 'numpy', 'pandas', 'scipy', 'matplotlib', 'torch', 'tensorflow']
    versions = {}
    for pkg in packages:
        try:
            module = import_module(pkg)
            version = getattr(module, '__version__', 'unknown')
            versions[pkg] = version
        except ImportError:
            versions[pkg] = "not installed"
    return versions

def list_kernels():
    """List available Jupyter kernels with details."""
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list", "--json"],
            check=True,
            capture_output=True,
            text=True
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None

def get_kernel_processes():
    """Get running kernel processes with PIDs."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            check=True,
            capture_output=True,
            text=True
        )
        kernel_procs = []
        for line in result.stdout.split('\n'):
            if 'ipykernel' in line and 'python' in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        kernel_procs.append({'pid': pid, 'line': line})
                    except ValueError:
                        pass
        return kernel_procs
    except subprocess.CalledProcessError:
        return []

def print_separator(title=""):
    """Print a formatted separator."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    else:
        print(f"\n{'-'*70}")

def print_menu(options):
    """Print a formatted menu and return user selection."""
    print("\nSelect an option:")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print(f"  0. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (0-{}): ".format(len(options))).strip()
            choice = int(choice)
            if 0 <= choice <= len(options):
                return choice
            else:
                print(f"Invalid choice. Please enter 0-{len(options)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

def kill_kernel_processes(pids=None):
    """Kill kernel processes by PID. If pids is None, kills all found."""
    procs = get_kernel_processes()
    if not procs:
        print("  No running kernel processes found")
        return

    targets = [p for p in procs if pids is None or p['pid'] in pids]
    for proc in targets:
        pid = proc['pid']
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"  ✓ Sent SIGTERM to kernel process {pid}")
        except ProcessLookupError:
            print(f"  - Process {pid} already gone")
        except PermissionError:
            print(f"  ✗ No permission to kill process {pid}")

def reinstall_kernel(kernel_name):
    """Reinstall a kernel."""
    try:
        print(f"\n  Reinstalling kernel: {kernel_name}...")

        # Remove first if exists
        subprocess.run(
            ["jupyter", "kernelspec", "remove", kernel_name, "-f"],
            capture_output=True
        )

        # Always use the current Python executable so venv kernels reinstall correctly
        result = subprocess.run(
            [sys.executable, "-m", "ipykernel", "install", "--user", "--name", kernel_name],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"  ✓ Kernel '{kernel_name}' reinstalled successfully")
            return True
        else:
            print(f"  ✗ Failed to reinstall kernel: {result.stderr}")
            return False

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def show_kernel_processes():
    """Display running kernel processes."""
    print_separator("RUNNING KERNEL PROCESSES")

    procs = get_kernel_processes()
    if procs:
        for proc in procs[:10]:
            print(f"  PID {proc['pid']}: {proc['line'][:100]}")
        if len(procs) > 10:
            print(f"  ... and {len(procs) - 10} more")
    else:
        print("  No running kernel processes found")

def show_information(notebook_path=None):
    """Display system and kernel information."""

    # System Information
    print_separator("SYSTEM INFORMATION")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    # Package Versions
    print_separator("INSTALLED PACKAGES")
    versions = get_package_versions()
    for pkg, version in sorted(versions.items()):
        status = "✓" if version != "not installed" else "✗"
        print(f"  {status} {pkg:15} {version}")

    # Jupyter Kernels
    print_separator("JUPYTER KERNELS")
    kernels_info = list_kernels()

    if kernels_info is None:
        print("  ✗ Jupyter not found. Install with: pip install jupyter")
        return False

    kernel_names = []
    if "kernelspecs" in kernels_info:
        for idx, (name, kernel_info) in enumerate(kernels_info['kernelspecs'].items(), 1):
            print(f"\n  [{idx}] {name}")
            print(f"      Path: {kernel_info.get('resource_dir', 'N/A')}")
            print(f"      Display: {kernel_info.get('display_name', 'N/A')}")
            kernel_names.append(name)

    show_kernel_processes()

    return kernel_names

def interactive_mode(kernel_names):
    """Interactive menu for kernel management."""
    while True:
        print_separator("KERNEL MANAGEMENT")

        options = [
            "View system info and kernels",
            "Kill a hanging kernel process",
            "Kill all kernel processes",
            "Reinstall a kernel",
            "View restart instructions"
        ]

        choice = print_menu(options)

        if choice == 0:
            print("\nExiting...")
            break

        elif choice == 1:
            kernel_names = show_information() or kernel_names

        elif choice == 2:
            procs = get_kernel_processes()
            if not procs:
                print("  No running kernel processes found")
            else:
                print("\nRunning kernel processes:")
                for i, proc in enumerate(procs, 1):
                    print(f"  {i}. PID {proc['pid']}: {proc['line'][:80]}")
                try:
                    sel = int(input("Select process number to kill: ")) - 1
                    if 0 <= sel < len(procs):
                        pid = procs[sel]['pid']
                        confirm = input(f"Kill PID {pid}? (y/n): ").strip().lower()
                        if confirm == 'y':
                            kill_kernel_processes(pids=[pid])
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Invalid input")

        elif choice == 3:
            confirm = input("Kill ALL running kernel processes? (y/n): ").strip().lower()
            if confirm == 'y':
                kill_kernel_processes()

        elif choice == 4:
            if kernel_names:
                print(f"\nAvailable kernels to reinstall:")
                for i, name in enumerate(kernel_names, 1):
                    print(f"  {i}. {name}")
                try:
                    kernel_idx = int(input("Select kernel number: ")) - 1
                    if 0 <= kernel_idx < len(kernel_names):
                        reinstall_kernel(kernel_names[kernel_idx])
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Invalid input")
            else:
                print("No kernels available")

        elif choice == 5:
            show_restart_instructions()

def show_restart_instructions():
    """Show how to restart kernel."""
    print_separator("HOW TO RESTART KERNEL")

    print("\n  Method 1: VS Code UI (Recommended)")
    print("    • Click the circular restart icon in the notebook toolbar (top right)")
    print("    • Or click the kernel name badge (bottom right) and select 'Restart Kernel'")

    print("\n  Method 2: Keyboard Shortcut")
    print("    • Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)")
    print("    • Type: 'Restart Notebook Kernel'")
    print("    • Press Enter")

    print("\n  Method 3: Command Line (Jupyter)")
    print("    • jupyter kernelspec list              # List all kernels")
    print("    • jupyter kernelspec remove python3    # Remove a kernel spec")
    print("    • python -m ipykernel install --user   # Reinstall kernel")

    print("\n  Method 4: Using this script")
    print("    • Run: python tools/restart_kernel.py")
    print("    • Select option '2' to kill a specific hanging process")
    print("    • Select option '4' to reinstall a kernel spec")

def main():
    """Main function."""
    try:
        notebook = sys.argv[1] if len(sys.argv) > 1 else None

        print("\n" + "="*70)
        print("  JUPYTER KERNEL MANAGEMENT TOOL")
        print("="*70)

        kernel_names = show_information(notebook)

        if kernel_names is False:
            sys.exit(1)

        interactive_mode(kernel_names)

    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
