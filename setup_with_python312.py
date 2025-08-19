#!/usr/bin/env python3
"""Setup script that ensures Python 3.12 is used throughout."""

import os
import sys
import subprocess
import shutil


def find_python312():
    """Find Python 3.12 executable."""
    possible_names = ['python3.12', 'python3.12.3', 'python']
    
    for name in possible_names:
        try:
            result = subprocess.run([name, '--version'], 
                                  capture_output=True, text=True, check=True)
            if '3.12' in result.stdout:
                print(f"✅ Found Python 3.12: {name}")
                return name
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print("❌ Python 3.12 not found")
    print("Please install Python 3.12 or ensure it's in your PATH")
    sys.exit(1)


def create_virtual_environment(python_cmd):
    """Create virtual environment with Python 3.12."""
    if os.path.exists("venv"):
        print("🗑️  Removing existing virtual environment...")
        shutil.rmtree("venv")
    
    print("📦 Creating virtual environment with Python 3.12...")
    subprocess.run([python_cmd, "-m", "venv", "venv"], check=True)
    print("✅ Virtual environment created")


def install_requirements():
    """Install requirements using the virtual environment."""
    print("📦 Installing requirements...")
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = os.path.join("venv", "Scripts", "pip")
        python_path = os.path.join("venv", "Scripts", "python")
    else:  # Unix/Linux/macOS
        pip_path = os.path.join("venv", "bin", "pip")
        python_path = os.path.join("venv", "bin", "python")
    
    # Upgrade pip first
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    try:
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("Try installing manually:")
        print(f"  source venv/bin/activate")
        print(f"  pip install -r requirements.txt")
        sys.exit(1)


def setup_environment():
    """Set up environment file."""
    if not os.path.exists(".env"):
        print("📝 Creating .env file from template...")
        with open(".env.example", "r") as src, open(".env", "w") as dst:
            dst.write(src.read())
        print("✅ .env file created")
        print("⚠️  Please edit .env and add your GOOGLE_API_KEY")
    else:
        print("✅ .env file already exists")


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
    
    print("✅ Data directories created")


def verify_installation():
    """Verify the installation works."""
    print("🔍 Verifying installation...")
    
    if os.name == 'nt':  # Windows
        python_path = os.path.join("venv", "Scripts", "python")
    else:  # Unix/Linux/macOS
        python_path = os.path.join("venv", "bin", "python")
    
    try:
        # Test basic imports
        test_script = """
import sys
print(f"Python version: {sys.version}")

try:
    import langchain
    print("✅ LangChain imported successfully")
except ImportError as e:
    print(f"❌ LangChain import failed: {e}")

try:
    import faiss
    print("✅ FAISS imported successfully")
except ImportError as e:
    print(f"❌ FAISS import failed: {e}")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("✅ Google GenAI imported successfully")
except ImportError as e:
    print(f"❌ Google GenAI import failed: {e}")
"""
        
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        
        if "❌" in result.stdout:
            print("⚠️  Some imports failed, but basic setup is complete")
        else:
            print("✅ All imports successful!")
            
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Verification failed: {e}")
        print("But setup should still work")


def main():
    """Main setup function."""
    print("🔬 Research Assistant Setup (Python 3.12)")
    print("=" * 50)
    
    try:
        python_cmd = find_python312()
        create_virtual_environment(python_cmd)
        install_requirements()
        setup_environment()
        create_directories()
        verify_installation()
        
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your GOOGLE_API_KEY")
        print("2. Activate virtual environment:")
        if os.name == 'nt':
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("3. Run the assistant:")
        print("   python main.py")
        print("\nOr run directly with:")
        print("   python3.12 main.py")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
