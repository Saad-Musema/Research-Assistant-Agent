#!/usr/bin/env python3
"""Install missing packages for the Research Assistant."""

import subprocess
import sys
import os

def install_packages():
    """Install missing packages."""
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Not in a virtual environment. Activating venv...")
        if os.name == 'nt':  # Windows
            pip_path = os.path.join("venv", "Scripts", "pip")
        else:  # Unix/Linux/macOS
            pip_path = os.path.join("venv", "bin", "pip")
    else:
        pip_path = "pip"
    
    # Essential packages for Groq-based setup
    essential_packages = [
        "langchain-groq",
        "sentence-transformers",
        "faiss-cpu",
        "python-dotenv"
    ]
    
    print("Installing essential packages...")
    
    for package in essential_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([pip_path, "install", package], check=True)
            print(f"{package} installed")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
    
    print("\n Essential packages installed!")
    print("Now you can run: python3.12 main.py")

if __name__ == "__main__":
    install_packages()
