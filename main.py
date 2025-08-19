#!/usr/bin/env python3
"""
Research Assistant with FAISS Vector Database
Main entry point for the application.
"""

import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.cli.main import main

if __name__ == "__main__":
    main()
