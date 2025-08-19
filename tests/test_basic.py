"""Basic tests for the Research Assistant."""

import unittest
import os
import sys
import tempfile
import shutil

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.vector_db.faiss_manager import FAISSManager
from src.document_processing.loaders import DocumentLoader


class TestConfig(unittest.TestCase):
    """Test configuration."""
    
    def test_config_creation(self):
        """Test that config can be created."""
        config = Config()
        self.assertIsNotNone(config)
    
    def test_supported_formats(self):
        """Test that supported formats are defined."""
        formats = DocumentLoader.get_supported_formats()
        self.assertIn('.pdf', formats)
        self.assertIn('.txt', formats)


class TestDocumentLoader(unittest.TestCase):
    """Test document loader."""
    
    def test_supported_formats(self):
        """Test getting supported formats."""
        formats = DocumentLoader.get_supported_formats()
        self.assertIsInstance(formats, list)
        self.assertTrue(len(formats) > 0)
    
    def test_unsupported_file(self):
        """Test handling of unsupported file format."""
        with self.assertRaises(ValueError):
            DocumentLoader.load_document("test.xyz")


class TestFAISSManager(unittest.TestCase):
    """Test FAISS manager."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_creation(self):
        """Test that FAISS manager can be created."""
        # This test will fail without proper API key setup
        # In production, you'd mock the embeddings
        try:
            manager = FAISSManager(self.temp_dir)
            self.assertIsNotNone(manager)
        except Exception as e:
            # Expected if no API key is set
            self.assertIn("GOOGLE_API_KEY", str(e))


if __name__ == '__main__':
    unittest.main()
