#!/usr/bin/env python3
"""Run the Research Assistant with the correct Python version."""

import os
import sys
import subprocess


def find_python():
    """Find the correct Python executable."""
    # First try the virtual environment
    if os.name == 'nt':  # Windows
        venv_python = os.path.join("venv", "Scripts", "python.exe")
    else:  # Unix/Linux/macOS
        venv_python = os.path.join("venv", "bin", "python")
    
    if os.path.exists(venv_python):
        return venv_python
    
    # Fallback to system Python 3.12
    possible_names = ['python3.12', 'python3.12.3', 'python3', 'python']
    
    for name in possible_names:
        try:
            result = subprocess.run([name, '--version'], 
                                  capture_output=True, text=True, check=True)
            if '3.12' in result.stdout or '3.11' in result.stdout or '3.10' in result.stdout or '3.9' in result.stdout:
                return name
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print(" No suitable Python version found")
    print("Please run setup_with_python312.py first")
    sys.exit(1)


def main():
    """Run the main application."""
    python_cmd = find_python()
    
    # Pass all arguments to the main script
    args = [python_cmd, "main.py"] + sys.argv[1:]
    
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
