#!/usr/bin/env python3
"""Test script to check available Google AI models."""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

def test_groq_models():
    """Test available Groq models."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file")
        return

    print("🔍 Testing Groq models...")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")

    # Test different Groq model names
    models_to_test = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ]

    for model_name in models_to_test:
        try:
            print(f"\n🧪 Testing Groq model: {model_name}")

            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=model_name,
                groq_api_key=api_key,
                temperature=0
            )

            # Test with a simple prompt
            response = llm.invoke("Hello, respond with just 'OK'")
            print(f"✅ {model_name}: {response.content}")

        except Exception as e:
            print(f"❌ {model_name}: {str(e)}")


def test_google_models():
    """Test available Google AI models."""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        return

    print("🔍 Testing Google AI models...")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")

    # Test different model names
    models_to_test = [
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]

    for model_name in models_to_test:
        try:
            print(f"\n🧪 Testing Google model: {model_name}")

            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0
            )

            # Test with a simple prompt
            response = llm.invoke("Hello, respond with just 'OK'")
            print(f"✅ {model_name}: {response.content}")

        except Exception as e:
            print(f"❌ {model_name}: {str(e)}")

    # Test embeddings
    print(f"\n🧪 Testing embeddings model: models/embedding-001")
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Test embedding
        result = embeddings.embed_query("test")
        print(f"✅ Embeddings: Generated {len(result)} dimensions")

    except Exception as e:
        print(f"❌ Embeddings: {str(e)}")


def test_imports():
    """Test if all required packages are installed."""
    print("📦 Testing package imports...")
    
    packages = [
        ("langchain", "langchain"),
        ("langchain_community", "langchain_community"),
        ("langchain_google_genai", "langchain_google_genai"),
        ("langchain_groq", "langchain_groq"),
        ("faiss", "faiss"),
        ("dotenv", "python-dotenv"),
        ("PyMuPDF", "fitz"),
        ("numpy", "numpy")
    ]
    
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name}: {e}")


if __name__ == "__main__":
    print("🔬 Research Assistant - Model Testing")
    print("=" * 50)
    
    test_imports()
    print("\n" + "=" * 50)
    test_groq_models()
    print("\n" + "=" * 50)
    test_google_models()
