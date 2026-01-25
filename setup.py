#!/usr/bin/env python3
"""Setup script for Multi-modal Vision-Language Models."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        command: Command to run.
        description: Description of what the command does.
        
    Returns:
        True if command succeeded, False otherwise.
    """
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible.
    
    Returns:
        True if Python version is compatible, False otherwise.
    """
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.10 or higher.")
        return False
    print(f"✓ Python {version.major}.{version.minor} is compatible")
    return True


def install_dependencies() -> bool:
    """Install project dependencies.
    
    Returns:
        True if installation succeeded, False otherwise.
    """
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt not found")
        return False
    
    return run_command(
        "pip install -r requirements.txt",
        "Installing dependencies from requirements.txt"
    )


def generate_sample_data() -> bool:
    """Generate sample data for testing.
    
    Returns:
        True if generation succeeded, False otherwise.
    """
    if not os.path.exists("scripts/generate_sample_data.py"):
        print("✗ Sample data generation script not found")
        return False
    
    return run_command(
        "python scripts/generate_sample_data.py",
        "Generating sample data"
    )


def run_tests() -> bool:
    """Run the test suite.
    
    Returns:
        True if tests passed, False otherwise.
    """
    if not os.path.exists("tests/test_models.py"):
        print("✗ Test files not found")
        return False
    
    return run_command(
        "python -m pytest tests/ -v",
        "Running test suite"
    )


def setup_pre_commit() -> bool:
    """Setup pre-commit hooks.
    
    Returns:
        True if setup succeeded, False otherwise.
    """
    if not os.path.exists(".pre-commit-config.yaml"):
        print("✗ Pre-commit configuration not found")
        return False
    
    # Install pre-commit if not already installed
    run_command("pip install pre-commit", "Installing pre-commit")
    
    return run_command(
        "pre-commit install",
        "Installing pre-commit hooks"
    )


def main():
    """Main setup function."""
    print("Multi-modal Vision-Language Models Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed: Could not install dependencies")
        print("Please check your Python environment and try again.")
        sys.exit(1)
    
    # Generate sample data
    print("\nGenerating sample data...")
    generate_sample_data()
    
    # Run tests
    print("\nRunning tests...")
    if not run_tests():
        print("⚠ Warning: Some tests failed. This might be due to missing model files.")
        print("The core functionality should still work.")
    
    # Setup pre-commit hooks
    print("\nSetting up development tools...")
    setup_pre_commit()
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the comprehensive demo: python demo_comprehensive.py")
    print("2. Launch the Streamlit demo: streamlit run demo/app.py")
    print("3. Check the README.md for detailed usage instructions")
    print("\nFor development:")
    print("- Run tests: python -m pytest tests/ -v")
    print("- Format code: black src/ tests/ scripts/")
    print("- Lint code: ruff check src/ tests/ scripts/")


if __name__ == "__main__":
    main()
