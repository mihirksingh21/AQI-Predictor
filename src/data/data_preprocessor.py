"""
Data Preprocessor Module for AQI Prediction System
Cleans and prepares data for machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
from config.config import *

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
        
    def clean_air_quality_data(self, df):
        """
        Clean and preprocess air quality data
        """
        if df.empty:
            return df
        
        print("Cleaning air quality data...")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        pollutant_columns = [col for col in df.columns if col in POLLUTANTS]
        
        for col in pollutant_columns:
            if col in df.columns:
                # Remove outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                # Fill remaining missing values with median
                df[col] = df[col].fillna(df[col].median())
        
        # Create AQI index (simplified calculation)
        df = self.calculate_aqi(df)
        
        return df
    
    def calculate_aqi(self, df):
        """
        Calculate AQI from pollutant concentrations
        Simplified AQI calculation based on PM2.5 and PM10
        """
        if 'pm25' in df.columns and 'pm10' in df.columns:
            # Use PM2.5 as primary indicator (simplified)
            df['aqi'] = df['pm25'].apply(self.pm25_to_aqi)
        elif 'pm10' in df.columns:
            # Use PM10 if PM2.5 not available
            df['aqi'] = df['pm10'].apply(self.pm10_to_aqi)
        else:
            # If no PM data, use average of available pollutants
            pollutant_cols = [col for col in POLLUTANTS if col in df.columns]
            if pollutant_cols:
                df['aqi'] = df[pollutant_cols].mean(axis=1)
            else:
                df['aqi'] = 50  # Default moderate AQI
        
        return df
    
    def pm25_to_aqi(self, pm25):
        """Convert PM2.5 concentration to AQI"""
        if pd.isna(pm25):
            return 50
        
        if pm25 <= 12.0:
            return self.linear_scale(pm25, 0, 12.0, 0, 50)
        elif pm25 <= 35.4:
            return self.linear_scale(pm25, 12.1, 35.4, 51, 100)
        elif pm25 <= 55.4:
            return self.linear_scale(pm25, 35.5, 55.4, 101, 150)
        elif pm25 <= 150.4:
            return self.linear_scale(pm25, 55.5, 150.4, 151, 200)
        elif pm25 <= 250.4:
            return self.linear_scale(pm25, 150.5, 250.4, 201, 300)
        else:
            return self.linear_scale(pm25, 250.5, 500.4, 301, 500)
    
    def pm10_to_aqi(self, pm10):
        """Convert PM10 concentration to AQI"""
        if pd.isna(pm10):
            return 50
        
        if pm10 <= 54:
            return self.linear_scale(pm10, 0, 54, 0, 50)
        elif pm10 <= 154:
            return self.linear_scale(pm10, 55, 154, 51, 100)
        elif pm10 <= 254:
            return self.linear_scale(pm10, 155, 254, 101, 150)
        elif pm10 <= 354:
            return self.linear_scale(pm10, 255, 354, 151, 200)
        elif pm10 <= 424:
            return self.linear_scale(pm10, 355, 424, 201, 300)
        else:
            return self.linear_scale(pm10, 425, 604, 301, 500)
    
    def linear_scale(self, value, low_val, high_val, low_aqi, high_aqi):
        """Linear interpolation for AQI calculation"""
        return ((high_aqi - low_aqi) / (high_val - low_val)) * (value - low_val) + low_aqi
    
    def clean_weather_data(self, df):
        """
        Clean and preprocess weather data
        """
        if df.empty:
            return df
        
        print("Cleaning weather data...")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Encode categorical variables
        categorical_columns = ['weather_main', 'weather_description']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                self.label_encoders[col] = le
        
        return df
    
    def create_time_features(self, df):
        """
        Create time-based features from datetime
        """
        if 'datetime' not in df.columns:
            return df
        
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def merge_datasets(self, aqi_df, weather_df):
        """
        Merge air quality and weather datasets
        """
        if aqi_df.empty or weather_df.empty:
            print("Cannot merge empty datasets")
            return pd.DataFrame()
        
        print("Merging datasets...")
        
        # Ensure datetime is the index for both datasets
        aqi_df = aqi_df.set_index('datetime')
        weather_df = weather_df.set_index('datetime')
        
        # Merge on datetime
        merged_df = pd.merge(
            aqi_df, 
            weather_df, 
            left_index=True, 
            right_index=True, 
            how='outer'
        )
        
        # Sort by datetime
        merged_df = merged_df.sort_index()
        
        return merged_df
    
    def prepare_features(self, df):
        """
        Prepare features for machine learning models
        """
        if df.empty:
            return None, None
        
        print("Preparing features...")
        
        # Select feature columns
        feature_columns = []
        
        # Air quality features
        pollutant_cols = [col for col in POLLUTANTS if col in df.columns]
        feature_columns.extend(pollutant_cols)
        
        # Weather features
        weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_deg', 'clouds']
        feature_columns.extend([col for col in weather_cols if col in df.columns])
        
        # Time features
        time_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend']
        feature_columns.extend([col for col in time_cols if col in df.columns])
        
        # Encoded weather features
        encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
        feature_columns.extend(encoded_cols)
        
        # Remove any columns that don't exist
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        if not feature_columns:
            print("No features found!")
            return None, None
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['aqi'] if 'aqi' in df.columns else None
        
        # Handle missing values in features
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        self.imputers['features'] = imputer
        
        # Handle missing values in target
        if y is not None:
            y = y.fillna(y.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.scalers['features'] = scaler
        
        return X_scaled, y
    
    def create_sequences(self, X, y, sequence_length=24):
        """
        Create sequences for LSTM models
        """
        if X is None or y is None:
            return None, None
        
        print(f"Creating sequences with length {sequence_length}...")
        
        # Remove any rows with NaN values
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_clean)):
            X_sequences.append(X_clean.iloc[i-sequence_length:i].values)
            y_sequences.append(y_clean.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """
        Split data into train, validation, and test sets
        """
        if X is None or y is None:
            return None, None, None, None, None, None
        
        total_samples = len(X)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        # For time series data, we split chronologically
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_all(self, aqi_data, weather_data, satellite_data=None):
        """
        Complete preprocessing pipeline
        """
        print("Starting complete preprocessing pipeline...")
        
        # Clean individual datasets
        aqi_clean = self.clean_air_quality_data(aqi_data)
        weather_clean = self.clean_weather_data(weather_data)
        
        # Create time features
        aqi_clean = self.create_time_features(aqi_clean)
        weather_clean = self.create_time_features(weather_clean)
        
        # Merge datasets
        merged_data = self.merge_datasets(aqi_clean, weather_clean)
        
        if merged_data.empty:
            print("No data after merging!")
            return None
        
        # Prepare features
        X, y = self.prepare_features(merged_data)
        
        if X is None:
            print("No features prepared!")
            return None
        
        # Create sequences for LSTM
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_seq, y_seq)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'merged_data': merged_data,
            'satellite_data': satellite_data
        }

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    print("Data preprocessor initialized successfully!") 