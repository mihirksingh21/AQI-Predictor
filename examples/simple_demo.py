"""
Simple Demo for AQI Prediction System
Works with minimal dependencies for testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def create_demo_data():
    """Create synthetic demo data"""
    print("Creating demo data...")
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')
    n_samples = len(dates)
    
    # Create synthetic air quality data
    np.random.seed(42)
    demo_data = pd.DataFrame({
        'datetime': dates,
        'pm25': np.random.normal(25, 10, n_samples),
        'pm10': np.random.normal(50, 20, n_samples),
        'no2': np.random.normal(30, 15, n_samples),
        'temperature': np.random.normal(25, 10, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'pressure': np.random.normal(1013, 20, n_samples),
        'wind_speed': np.random.uniform(0, 20, n_samples)
    })
    
    # Calculate AQI (simplified)
    demo_data['aqi'] = demo_data['pm25'].apply(lambda x: max(0, min(500, x * 2)))
    
    return demo_data

def simple_aqi_prediction(data):
    """Simple AQI prediction using basic statistics"""
    print("Running simple AQI prediction...")
    
    # Simple feature engineering
    data['hour'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Select features
    features = ['pm25', 'pm10', 'no2', 'temperature', 'humidity', 'pressure', 'wind_speed', 'hour', 'is_weekend']
    X = data[features].fillna(method='ffill')
    y = data['aqi']
    
    # Simple prediction (using moving average)
    window_size = 24
    predictions = y.rolling(window=window_size, center=True).mean().fillna(method='bfill')
    
    # Calculate simple metrics
    mse = np.mean((y - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - predictions))
    
    print(f"Simple Prediction Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return predictions, {'rmse': rmse, 'mae': mae}

def create_simple_visualization(data, predictions):
    """Create simple visualization"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot actual vs predicted
        plt.subplot(2, 1, 1)
        plt.plot(data['datetime'], data['aqi'], label='Actual AQI', alpha=0.7)
        plt.plot(data['datetime'], predictions, label='Predicted AQI', alpha=0.7)
        plt.title('AQI Prediction Demo')
        plt.xlabel('Time')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot AQI distribution
        plt.subplot(2, 1, 2)
        plt.hist(data['aqi'], bins=30, alpha=0.7, color='orange')
        plt.title('AQI Distribution')
        plt.xlabel('AQI')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/simple_demo.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved to results/simple_demo.png")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")

def get_aqi_category(aqi_value):
    """Get AQI category"""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def main():
    """Main demo function"""
    print("=" * 50)
    print("AQI PREDICTION SYSTEM - SIMPLE DEMO")
    print("=" * 50)
    
    # Create demo data
    data = create_demo_data()
    print(f"✓ Created {len(data)} data points")
    
    # Run simple prediction
    predictions, metrics = simple_aqi_prediction(data)
    
    # Show some predictions
    print("\nSample Predictions:")
    print("Time\t\t\tActual AQI\tPredicted AQI\tCategory")
    print("-" * 70)
    
    for i in range(0, len(data), 24):  # Show every 24th hour
        actual = data.iloc[i]['aqi']
        predicted = predictions.iloc[i]
        category = get_aqi_category(predicted)
        time_str = data.iloc[i]['datetime'].strftime('%Y-%m-%d %H:%M')
        print(f"{time_str}\t{actual:.1f}\t\t{predicted:.1f}\t\t{category}")
    
    # Create visualization
    create_simple_visualization(data, predictions)
    
    print("\n" + "=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\nThis demo shows basic AQI prediction functionality.")
    print("For full features, install all dependencies and run main.py")

if __name__ == "__main__":
    main() 