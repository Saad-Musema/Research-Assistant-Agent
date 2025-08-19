#!/usr/bin/env python3
"""Setup script for the Research Assistant."""

import os
import sys
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("Python 3.9 or higher is required")
        print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}")
        print("   LangChain and related packages require Python 3.9+")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")


def create_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.run(["python3.12", "-m", "venv", "venv"], check=True)
        print("Virtual environment created")
    else:
        print("Virtual environment already exists")


def install_requirements():
    """Install requirements."""
    print("Installing requirements...")
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:  # Unix/Linux/macOS
        pip_path = os.path.join("venv", "bin", "pip")
    
    try:
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("Failed to install requirements")
        print("Try running manually:")
        print(f"  {pip_path} install -r requirements.txt")


def setup_environment():
    """Set up environment file."""
    if not os.path.exists(".env"):
        print("Creating .env file from template...")
        with open(".env.example", "r") as src, open(".env", "w") as dst:
            dst.write(src.read())
        print(".env file created")
        print("Please edit .env and add your GOOGLE_API_KEY")
    else:
        print(".env file already exists")


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/books",
        "data/papers", 
        "data/vector_db",
        "data/cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Data directories created")


def main():
    """Main setup function."""
    print("Research Assistant Setup")
    print("=" * 40)
    
    try:
        check_python_version()
        create_virtual_environment()
        install_requirements()
        setup_environment()
        create_directories()
        
        print("\nSetup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your GOOGLE_API_KEY")
        print("2. Activate virtual environment:")
        if os.name == 'nt':
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("3. Run the assistant:")
        print("   python main.py")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
