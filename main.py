#!/usr/bin/env python3
"""
AQI Prediction System - Main Entry Point

This is the main entry point for the AQI Prediction System.
It provides a command-line interface to run the complete pipeline.
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the core modules directly
from src.core.main import AQIPredictionSystem
from src.core.main_no_tensorflow import AQIPredictionSystemNoTF

def main():
    """Main entry point for the AQI Prediction System."""
    print("=" * 60)
    print("🌬️ AQI PREDICTION SYSTEM")
    print("=" * 60)
    print("Advanced Air Quality Index Prediction using Machine Learning")
    print("and Satellite Data")
    print("=" * 60)
    
    print("\nChoose your system:")
    print("1. Full System (with TensorFlow)")
    print("2. Traditional ML Only (no TensorFlow)")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting Full AQI Prediction System...")
                system = AQIPredictionSystem()
                system.run()
                
            elif choice == "2":
                print("\n🤖 Starting Traditional ML AQI Prediction System...")
                system = AQIPredictionSystemNoTF()
                system.run()
                
            elif choice == "3":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please check your configuration and try again.")

if __name__ == "__main__":
    main() 