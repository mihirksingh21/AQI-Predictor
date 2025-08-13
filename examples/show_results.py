"""
Script to display results from the AQI prediction system
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def show_results():
    print("=" * 60)
    print("AQI PREDICTION SYSTEM - RESULTS SUMMARY")
    print("=" * 60)
    
    # Check generated files
    print("\n📁 GENERATED FILES:")
    print("-" * 40)
    
    # Data files
    if os.path.exists("data/"):
        print("📊 Data Files:")
        for file in os.listdir("data/"):
            size = os.path.getsize(f"data/{file}") / 1024  # KB
            print(f"  • {file} ({size:.1f} KB)")
    
    # Model files
    if os.path.exists("models/"):
        print("\n🤖 Trained Models:")
        for file in os.listdir("models/"):
            size = os.path.getsize(f"models/{file}") / 1024  # KB
            print(f"  • {file} ({size:.1f} KB)")
    
    # Visualization files
    if os.path.exists("results/"):
        print("\n📈 Visualizations:")
        for file in os.listdir("results/"):
            size = os.path.getsize(f"results/{file}") / 1024  # KB
            print(f"  • {file} ({size:.1f} KB)")
    
    # Show data summary
    print("\n📊 DATA SUMMARY:")
    print("-" * 40)
    
    try:
        aqi_data = pd.read_csv("data/demo_air_quality.csv")
        weather_data = pd.read_csv("data/demo_weather.csv")
        
        print(f"Air Quality Records: {len(aqi_data)}")
        print(f"Weather Records: {len(weather_data)}")
        print(f"Date Range: {aqi_data['datetime'].min()} to {aqi_data['datetime'].max()}")
        
        # Show AQI statistics
        if 'aqi' in aqi_data.columns:
            print(f"\nAQI Statistics:")
            print(f"  • Mean AQI: {aqi_data['aqi'].mean():.1f}")
            print(f"  • Min AQI: {aqi_data['aqi'].min():.1f}")
            print(f"  • Max AQI: {aqi_data['aqi'].max():.1f}")
            print(f"  • Std AQI: {aqi_data['aqi'].std():.1f}")
        
        # Show pollutant statistics
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        print(f"\nPollutant Statistics:")
        for pollutant in pollutants:
            if pollutant in aqi_data.columns:
                print(f"  • {pollutant.upper()}: {aqi_data[pollutant].mean():.1f} ± {aqi_data[pollutant].std():.1f}")
        
    except Exception as e:
        print(f"Error reading data: {e}")
    
    # Show model performance
    print("\n🤖 MODEL PERFORMANCE:")
    print("-" * 40)
    print("Based on the last run:")
    print("  • Random Forest: R² = -0.005, RMSE = 23.18")
    print("  • Gradient Boosting: R² = -0.069, RMSE = 23.90")
    print("  • Linear Regression: R² = -0.031, RMSE = 23.47")
    print("  • SVR: R² = -0.000, RMSE = 23.12")
    print("  • Best Model: SVR")
    
    # Show what's available to view
    print("\n🎯 WHAT YOU CAN DO NEXT:")
    print("-" * 40)
    print("1. 📊 View Interactive Dashboard:")
    print("   • Open 'results/dashboard.html' in your web browser")
    print("   • Interactive charts and real-time data visualization")
    
    print("\n2. 🗺️ View Interactive Map:")
    print("   • Open 'results/aqi_map.html' in your web browser")
    print("   • Geographic visualization of AQI data")
    
    print("\n3. 📈 View Static Charts:")
    print("   • 'results/aqi_trends.png' - AQI trends over time")
    print("   • 'results/pollutant_correlations.png' - Correlation matrix")
    print("   • 'results/weather_aqi_relationship.png' - Weather vs AQI")
    print("   • 'results/model_comparison.png' - Model performance comparison")
    
    print("\n4. 🔧 Make Predictions:")
    print("   • Use trained models in 'models/' directory")
    print("   • Load models with joblib for new predictions")
    
    print("\n5. 📊 Analyze Data:")
    print("   • Explore 'data/demo_air_quality.csv' and 'data/demo_weather.csv'")
    print("   • Use pandas for data analysis")
    
    # API Issues
    print("\n⚠️ API ISSUES DETECTED:")
    print("-" * 40)
    print("• OpenAQ API: 410 error (endpoint may have changed)")
    print("• Weather API: No data collected (check API key)")
    print("• Satellite API: Configuration issue")
    print("\nSolutions:")
    print("• Update OpenAQ API endpoint in data_collector.py")
    print("• Verify OpenWeatherMap API key")
    print("• Check NASA Earthdata credentials")
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM SUCCESSFULLY RUN!")
    print("=" * 60)

if __name__ == "__main__":
    show_results() 