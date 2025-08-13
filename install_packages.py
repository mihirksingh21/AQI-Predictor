"""
Alternative package installation script for AQI Prediction System
Installs packages one by one to avoid dependency conflicts
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def install_core_packages():
    """Install core packages first"""
    core_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "Pillow>=8.0.0",
        "joblib>=1.1.0"
    ]
    
    print("Installing core packages...")
    for package in core_packages:
        if not install_package(package):
            print(f"Warning: Failed to install {package}")
    
    return True

def install_ml_packages():
    """Install machine learning packages"""
    ml_packages = [
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    print("\nInstalling machine learning packages...")
    for package in ml_packages:
        if not install_package(package):
            print(f"Warning: Failed to install {package}")
    
    return True

def install_dl_packages():
    """Install deep learning packages"""
    print("\nInstalling deep learning packages...")
    
    # Try to install TensorFlow
    if not install_package("tensorflow>=2.8.0"):
        print("Warning: TensorFlow installation failed. Trying CPU-only version...")
        if not install_package("tensorflow-cpu>=2.8.0"):
            print("Warning: TensorFlow installation failed. You may need to install it manually.")
    
    return True

def install_visualization_packages():
    """Install visualization packages"""
    viz_packages = [
        "plotly>=5.0.0",
        "folium>=0.12.0"
    ]
    
    print("\nInstalling visualization packages...")
    for package in viz_packages:
        if not install_package(package):
            print(f"Warning: Failed to install {package}")
    
    return True

def install_optional_packages():
    """Install optional packages (can fail without breaking the system)"""
    optional_packages = [
        "rasterio>=1.2.0",
        "earthaccess>=0.6.0",
        "opencv-python>=4.5.0",
        "geopandas>=0.10.0",
        "osmnx>=1.1.0",
        "geopy>=2.2.0"
    ]
    
    print("\nInstalling optional packages...")
    for package in optional_packages:
        if not install_package(package):
            print(f"Note: {package} is optional and not critical for basic functionality")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = ["data", "models", "results", "images"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}/ directory")
    
    return True

def test_imports():
    """Test if core packages can be imported"""
    print("\nTesting package imports...")
    
    core_imports = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("requests", "requests"),
        ("PIL", "Pillow")
    ]
    
    failed_imports = []
    
    for import_name, package_name in core_imports:
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Import failed")
            failed_imports.append(package_name)
    
    if failed_imports:
        print(f"\nWarning: Some packages failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✓ All core packages imported successfully!")
    return True

def main():
    """Main installation function"""
    print("=" * 60)
    print("AQI PREDICTION SYSTEM - ALTERNATIVE INSTALLATION")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Install packages in order
    install_core_packages()
    install_ml_packages()
    install_dl_packages()
    install_visualization_packages()
    install_optional_packages()
    
    # Test imports
    test_imports()
    
    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Choose option 3 for demo mode to test the system")
    print("3. If you encounter import errors, install missing packages manually")
    print("\nFor manual installation of specific packages:")
    print("pip install package_name")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 