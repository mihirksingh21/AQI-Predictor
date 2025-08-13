"""
Main AQI Prediction System (No TensorFlow Version)
Complete pipeline for air quality prediction using traditional ML models
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_collector import DataCollector
from data_preprocessor import DataPreprocessor
from visualization import AQIVisualizer
from config import *

class AQIPredictionSystemNoTF:
    def __init__(self):
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.visualizer = AQIVisualizer()
        
    def create_traditional_models(self):
        """Create traditional ML models without TensorFlow"""
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.svm import SVR
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import joblib
            
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'linear_regression': LinearRegression(),
                'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
            }
            
            return models, {
                'mean_squared_error': mean_squared_error,
                'mean_absolute_error': mean_absolute_error,
                'r2_score': r2_score,
                'joblib': joblib
            }
            
        except ImportError as e:
            print(f"Error importing sklearn: {e}")
            return None, None
    
    def train_and_evaluate_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train and evaluate traditional ML models"""
        models, metrics = self.create_traditional_models()
        
        if models is None:
            print("Failed to create models")
            return {}
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Reshape data for traditional ML (remove sequence dimension if present)
            if len(X_train.shape) == 3:
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
            else:
                X_train_flat = X_train
                X_val_flat = X_val
                X_test_flat = X_test
            
            # Train model
            model.fit(X_train_flat, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_flat)
            
            # Calculate metrics
            mse = metrics['mean_squared_error'](y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = metrics['mean_absolute_error'](y_test, y_pred)
            r2 = metrics['r2_score'](y_test, y_pred)
            
            results[name] = ({
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }, y_pred)
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
            
            # Save model
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
            metrics['joblib'].dump(model, model_path)
            print(f"Model saved to {model_path}")
        
        return results
    
    def run_complete_pipeline(self, city=DEFAULT_CITY, days=HISTORICAL_DAYS):
        """
        Run the complete AQI prediction pipeline (without TensorFlow)
        """
        print("=" * 60)
        print("AQI PREDICTION SYSTEM - TRADITIONAL ML PIPELINE")
        print("=" * 60)
        print(f"City: {city}")
        print(f"Historical days: {days}")
        print(f"Start time: {datetime.now()}")
        print("=" * 60)
        
        # Step 1: Data Collection
        print("\nSTEP 1: DATA COLLECTION")
        print("-" * 30)
        data = self.collector.collect_all_data(city, days)
        
        if not data['aqi_data'].empty or not data['weather_data'].empty:
            print("✓ Data collection completed successfully")
        else:
            print("⚠ Limited data available - proceeding with available data")
        
        # Step 2: Data Preprocessing
        print("\nSTEP 2: DATA PREPROCESSING")
        print("-" * 30)
        processed_data = self.preprocessor.preprocess_all(
            data['aqi_data'], 
            data['weather_data'], 
            data['satellite_data']
        )
        
        if processed_data is None:
            print("❌ Data preprocessing failed - insufficient data")
            return None
        
        print("✓ Data preprocessing completed successfully")
        
        # Step 3: Model Training (Traditional ML only)
        print("\nSTEP 3: MODEL TRAINING (Traditional ML)")
        print("-" * 30)
        results = self.train_and_evaluate_models(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            processed_data['X_test'],
            processed_data['y_test']
        )
        
        if results:
            print("✓ Model training completed successfully")
        else:
            print("❌ Model training failed")
            return None
        
        # Step 4: Visualization
        print("\nSTEP 4: VISUALIZATION")
        print("-" * 30)
        self.visualizer.create_all_visualizations(
            processed_data['merged_data'],
            results
        )
        
        print("✓ Visualizations created successfully")
        
        # Step 5: Results Summary
        print("\nSTEP 5: RESULTS SUMMARY")
        print("-" * 30)
        self.print_results_summary(results, processed_data)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"End time: {datetime.now()}")
        print("=" * 60)
        
        return {
            'data': data,
            'processed_data': processed_data,
            'results': results
        }
    
    def print_results_summary(self, results, processed_data):
        """
        Print summary of model results
        """
        print("\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        # Find best model
        best_model = None
        best_r2 = -float('inf')
        
        for model_name, (metrics, predictions) in results.items():
            r2 = metrics['r2']
            rmse = metrics['rmse']
            mae = metrics['mae']
            
            print(f"{model_name.upper():<20} | R²: {r2:.3f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model_name
        
        print("-" * 40)
        print(f"BEST MODEL: {best_model.upper()} (R²: {best_r2:.3f})")
        
        # Data summary
        print(f"\nDATA SUMMARY:")
        print(f"Total samples: {len(processed_data['merged_data'])}")
        print(f"Training samples: {len(processed_data['X_train'])}")
        print(f"Validation samples: {len(processed_data['X_val'])}")
        print(f"Test samples: {len(processed_data['X_test'])}")
        print(f"Features: {len(processed_data['feature_names'])}")
        
        # Feature importance (for tree-based models)
        if 'random_forest' in results:
            try:
                import joblib
                rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
                feature_importance = pd.DataFrame({
                    'feature': processed_data['feature_names'],
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\nTOP 10 FEATURES (Random Forest):")
                for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                    print(f"{i+1:2d}. {row['feature']:<20} | {row['importance']:.3f}")
            except:
                print("Could not load feature importance")
    
    def create_demo_data(self):
        """
        Create demo data for testing when real data is not available
        """
        print("Creating demo data for testing...")
        
        # Generate synthetic air quality data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h')
        n_samples = len(dates)
        
        # Synthetic pollutant data
        np.random.seed(42)
        demo_aqi = pd.DataFrame({
            'datetime': dates,
            'pm25': np.random.normal(25, 10, n_samples),
            'pm10': np.random.normal(50, 20, n_samples),
            'no2': np.random.normal(30, 15, n_samples),
            'so2': np.random.normal(10, 5, n_samples),
            'co': np.random.normal(1.5, 0.8, n_samples),
            'o3': np.random.normal(40, 20, n_samples),
            'location': 'Demo Station',
            'coordinates.latitude': 28.6139,
            'coordinates.longitude': 77.2090
        })
        
        # Synthetic weather data
        demo_weather = pd.DataFrame({
            'datetime': dates,
            'temperature': np.random.normal(25, 10, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'pressure': np.random.normal(1013, 20, n_samples),
            'wind_speed': np.random.uniform(0, 20, n_samples),
            'wind_deg': np.random.uniform(0, 360, n_samples),
            'clouds': np.random.uniform(0, 100, n_samples),
            'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain'], n_samples),
            'weather_description': np.random.choice(['clear sky', 'scattered clouds', 'light rain'], n_samples)
        })
        
        # Save demo data
        os.makedirs(DATA_DIR, exist_ok=True)
        demo_aqi.to_csv(f"{DATA_DIR}/demo_air_quality.csv", index=False)
        demo_weather.to_csv(f"{DATA_DIR}/demo_weather.csv", index=False)
        
        print("Demo data created successfully!")
        
        return {
            'aqi_data': demo_aqi,
            'weather_data': demo_weather,
            'satellite_data': None
        }

def main():
    """
    Main function to run the AQI prediction system (no TensorFlow)
    """
    system = AQIPredictionSystemNoTF()
    
    print("AQI PREDICTION SYSTEM (Traditional ML Only)")
    print("=" * 50)
    print("1. Run complete pipeline")
    print("2. Create demo data and run pipeline")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        city = input("Enter city name (default: Delhi): ").strip() or DEFAULT_CITY
        days = input("Enter number of historical days (default: 30): ").strip()
        days = int(days) if days.isdigit() else HISTORICAL_DAYS
        
        system.run_complete_pipeline(city, days)
        
    elif choice == '2':
        print("\nCreating demo data and running pipeline...")
        demo_data = system.create_demo_data()
        
        # Use demo data for preprocessing
        processed_data = system.preprocessor.preprocess_all(
            demo_data['aqi_data'],
            demo_data['weather_data'],
            demo_data['satellite_data']
        )
        
        if processed_data:
            results = system.train_and_evaluate_models(
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_val'],
                processed_data['y_val'],
                processed_data['X_test'],
                processed_data['y_test']
            )
            
            system.visualizer.create_all_visualizations(
                processed_data['merged_data'],
                results
            )
            
            system.print_results_summary(results, processed_data)
        else:
            print("Failed to process demo data")
    
    elif choice == '3':
        print("Exiting...")
        sys.exit(0)
    
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main() 