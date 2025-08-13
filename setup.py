"""
Setup script for AQI Prediction System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    
    directories = ["data", "models", "results", "images"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}/ directory")
    
    return True

def check_dependencies():
    """Check if all dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = [
        "numpy", "pandas", "scikit-learn", "tensorflow", 
        "matplotlib", "seaborn", "requests", "Pillow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        return False
    
    print("✓ All dependencies are available!")
    return True

def main():
    """Main setup function"""
    print("=" * 50)
    print("AQI PREDICTION SYSTEM - SETUP")
    print("=" * 50)
    
    # Create directories
    if not create_directories():
        print("Failed to create directories")
        return
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("Some dependencies are missing. Please install them manually.")
        return
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Update config.py with your API keys (optional)")
    print("2. Run: python main.py")
    print("3. Choose option 3 for demo mode to test the system")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 