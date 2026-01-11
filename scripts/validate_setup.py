#!/usr/bin/env python3
"""
AI4Org Setup Validation Script

This script validates that your environment is properly configured
to run the AI4Org application.
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_check(passed, message):
    """Print a check result"""
    symbol = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    print(f"{symbol} [{status}] {message}")
    return passed

def check_python_version():
    """Check if Python version is 3.10+"""
    print_header("Python Version Check")
    version = sys.version_info
    required = (3, 10)
    passed = version >= required
    print_check(
        passed,
        f"Python {version.major}.{version.minor}.{version.micro} "
        f"(required: {required[0]}.{required[1]}+)"
    )
    return passed

def check_packages():
    """Check if required packages are installed"""
    print_header("Package Installation Check")
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sklearn", "scikit-learn"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS (CPU)"),
        ("dotenv", "python-dotenv"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
    ]
    
    all_passed = True
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print_check(True, f"{display_name} installed")
        except ImportError:
            print_check(False, f"{display_name} NOT installed")
            all_passed = False
    
    return all_passed

def check_optional_packages():
    """Check optional packages"""
    print_header("Optional Package Check")
    
    # Check PyWebView for frontend
    try:
        import webview
        print_check(True, "PyWebView installed (frontend will work)")
    except ImportError:
        print_check(False, "PyWebView NOT installed (frontend won't work)")
    
    # Check for GPU support
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_check(True, f"CUDA available: {gpu_count} GPU(s) - {gpu_name}")
        else:
            print_check(False, "CUDA not available (will use CPU)")
    except Exception as e:
        print_check(False, f"Error checking CUDA: {e}")

def check_data_files():
    """Check if required data files exist"""
    print_header("Data Files Check")
    
    project_root = Path(__file__).parent.parent
    
    files = [
        ("data/processed/corpus.txt", "Corpus file"),
        ("data/qa/qa.json", "QA pairs file"),
    ]
    
    all_passed = True
    for file_path, description in files:
        full_path = project_root / file_path
        exists = full_path.exists()
        size = full_path.stat().st_size if exists else 0
        
        if exists:
            print_check(True, f"{description}: {full_path} ({size:,} bytes)")
        else:
            print_check(False, f"{description}: {full_path} NOT FOUND")
            all_passed = False
    
    return all_passed

def check_environment_variables():
    """Check environment variables"""
    print_header("Environment Variables Check")
    
    # Check if .env file exists
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if env_file.exists():
        print_check(True, f".env file exists: {env_file}")
    else:
        print_check(False, f".env file NOT found (using defaults)")
        if env_example.exists():
            print(f"  → Copy {env_example} to {env_file} and customize")
    
    # Check some important env vars
    from dotenv import load_dotenv
    load_dotenv()
    
    vars_to_check = [
        ("ADMIN_PIN", "Admin PIN for frontend"),
        ("GEN_MODEL", "Generator model name"),
        ("DEVICE", "Device configuration"),
    ]
    
    for var_name, description in vars_to_check:
        value = os.getenv(var_name)
        if value:
            # Mask sensitive values
            display_value = "***" if "PIN" in var_name or "KEY" in var_name else value
            print_check(True, f"{var_name}={display_value} ({description})")
        else:
            print_check(False, f"{var_name} not set ({description})")

def check_model_cache():
    """Check if models are cached"""
    print_header("Model Cache Check")
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if cache_dir.exists():
        cached_models = list(cache_dir.glob("models--*"))
        print_check(True, f"HuggingFace cache exists: {len(cached_models)} models cached")
        
        # Check for specific models
        models_to_check = [
            "TinyLlama",
            "distilbert",
        ]
        
        for model_name in models_to_check:
            found = any(model_name.lower() in str(m).lower() for m in cached_models)
            if found:
                print(f"  → {model_name} appears to be cached")
    else:
        print_check(False, "HuggingFace cache not found (models will be downloaded on first run)")

def check_write_permissions():
    """Check write permissions for important directories"""
    print_header("Write Permissions Check")
    
    project_root = Path(__file__).parent.parent
    
    dirs_to_check = [
        "saved_models_improved",
        "data/raw",
        "logs",
    ]
    
    all_passed = True
    for dir_path in dirs_to_check:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Try to write a test file
        test_file = full_path / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print_check(True, f"Write access to {dir_path}/")
        except Exception as e:
            print_check(False, f"Cannot write to {dir_path}/: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all validation checks"""
    print("\n" + "="*60)
    print("  AI4Org Setup Validation")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("Data Files", check_data_files),
        ("Write Permissions", check_write_permissions),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ Error running {name} check: {e}")
            results[name] = False
    
    # Optional checks (don't affect overall status)
    try:
        check_optional_packages()
    except Exception as e:
        print(f"\n⚠ Error running optional checks: {e}")
    
    try:
        check_environment_variables()
    except Exception as e:
        print(f"\n⚠ Error checking environment variables: {e}")
    
    try:
        check_model_cache()
    except Exception as e:
        print(f"\n⚠ Error checking model cache: {e}")
    
    # Summary
    print_header("Validation Summary")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for name, passed in results.items():
        print_check(passed, name)
    
    print(f"\nPassed: {passed_count}/{total_count} checks")
    
    if all(results.values()):
        print("\n✓ All critical checks passed! You're ready to run AI4Org.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Train the model: python -m hallucination_reduction.main")
        print("  3. Run inference: python -m hallucination_reduction.inference")
        print("  4. Launch frontend: cd frontend && python main.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above before running AI4Org.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Ensure data files exist in data/processed/ and data/qa/")
        print("  - Check file permissions")
        return 1

if __name__ == "__main__":
    sys.exit(main())
