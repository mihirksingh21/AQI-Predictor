"""
Machine Learning Models for AQI Prediction System
Includes CNN-LSTM hybrid model and traditional ML models
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from config import *

class AQIPredictionModels:
    def __init__(self):
        self.models = {}
        self.history = {}
        self.scalers = {}
        
    def create_cnn_lstm_model(self, input_shape, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE):
        """
        Create CNN-LSTM hybrid model for AQI prediction
        """
        print("Creating CNN-LSTM hybrid model...")
        
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(dropout_rate),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(dropout_rate),
            
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(dropout_rate),
            
            # LSTM layers for temporal dependencies
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            
            LSTM(lstm_units // 2, return_sequences=False),
            Dropout(dropout_rate),
            
            # Dense layers for final prediction
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(1, activation='linear')  # Linear activation for regression
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def create_lstm_model(self, input_shape, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE):
        """
        Create LSTM model for AQI prediction
        """
        print("Creating LSTM model...")
        
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            
            LSTM(lstm_units // 2, return_sequences=True),
            Dropout(dropout_rate),
            
            LSTM(lstm_units // 4, return_sequences=False),
            Dropout(dropout_rate),
            
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def create_cnn_lstm_satellite_model(self, sequence_shape, satellite_shape, lstm_units=LSTM_UNITS):
        """
        Create CNN-LSTM model that incorporates satellite imagery
        """
        print("Creating CNN-LSTM model with satellite data...")
        
        # Input for time series data
        sequence_input = Input(shape=sequence_shape)
        
        # CNN layers for time series
        x1 = Conv1D(64, 3, activation='relu')(sequence_input)
        x1 = MaxPooling1D(2)(x1)
        x1 = Dropout(0.3)(x1)
        
        x1 = Conv1D(128, 3, activation='relu')(x1)
        x1 = MaxPooling1D(2)(x1)
        x1 = Dropout(0.3)(x1)
        
        # LSTM layers
        x1 = LSTM(lstm_units, return_sequences=True)(x1)
        x1 = Dropout(0.3)(x1)
        x1 = LSTM(lstm_units // 2, return_sequences=False)(x1)
        x1 = Dropout(0.3)(x1)
        
        # Input for satellite data
        satellite_input = Input(shape=satellite_shape)
        
        # CNN layers for satellite imagery
        x2 = Conv1D(32, 3, activation='relu')(satellite_input)
        x2 = MaxPooling1D(2)(x2)
        x2 = Dropout(0.3)(x2)
        
        x2 = Conv1D(64, 3, activation='relu')(x2)
        x2 = MaxPooling1D(2)(x2)
        x2 = Dropout(0.3)(x2)
        
        x2 = Flatten()(x2)
        x2 = Dense(64, activation='relu')(x2)
        x2 = Dropout(0.3)(x2)
        
        # Concatenate both inputs
        combined = Concatenate()([x1, x2])
        
        # Final dense layers
        x = Dense(128, activation='relu')(combined)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        output = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=[sequence_input, satellite_input], outputs=output)
        
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def create_traditional_models(self):
        """
        Create traditional machine learning models
        """
        print("Creating traditional ML models...")
        
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
        
        return models
    
    def train_deep_learning_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """
        Train deep learning model with callbacks
        """
        print(f"Training {model_name}...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models[model_name] = model
        self.history[model_name] = history
        
        return model, history
    
    def train_traditional_models(self, X_train, y_train, X_val, y_val):
        """
        Train traditional machine learning models
        """
        print("Training traditional ML models...")
        
        # Reshape data for traditional ML (remove sequence dimension)
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_flat = X_train
            X_val_flat = X_val
        
        traditional_models = self.create_traditional_models()
        
        for name, model in traditional_models.items():
            print(f"Training {name}...")
            model.fit(X_train_flat, y_train)
            self.models[name] = model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance
        """
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        if model_name in ['cnn_lstm', 'lstm', 'cnn_lstm_satellite']:
            y_pred = model.predict(X_test)
        else:
            # Reshape for traditional ML
            if len(X_test.shape) == 3:
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
            else:
                X_test_flat = X_test
            y_pred = model.predict(X_test_flat)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"{model_name} Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        return metrics, y_pred
    
    def save_model(self, model, model_name, save_dir=MODELS_DIR):
        """
        Save trained model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if model_name in ['cnn_lstm', 'lstm', 'cnn_lstm_satellite']:
            # Save TensorFlow model
            model_path = os.path.join(save_dir, f"{model_name}.h5")
            model.save(model_path)
        else:
            # Save scikit-learn model
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_name, load_dir=MODELS_DIR):
        """
        Load trained model
        """
        if model_name in ['cnn_lstm', 'lstm', 'cnn_lstm_satellite']:
            # Load TensorFlow model
            model_path = os.path.join(load_dir, f"{model_name}.h5")
            model = tf.keras.models.load_model(model_path)
        else:
            # Load scikit-learn model
            model_path = os.path.join(load_dir, f"{model_name}.pkl")
            model = joblib.load(model_path)
        
        return model
    
    def predict_aqi(self, model, X_input, model_name):
        """
        Make AQI predictions
        """
        if model_name in ['cnn_lstm', 'lstm', 'cnn_lstm_satellite']:
            prediction = model.predict(X_input)
        else:
            # Reshape for traditional ML
            if len(X_input.shape) == 3:
                X_input_flat = X_input.reshape(X_input.shape[0], -1)
            else:
                X_input_flat = X_input
            prediction = model.predict(X_input_flat)
        
        return prediction
    
    def get_aqi_category(self, aqi_value):
        """
        Get AQI category based on value
        """
        for (low, high), category in AQI_CATEGORIES.items():
            if low <= aqi_value <= high:
                return category
        return "Unknown"
    
    def train_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test, satellite_data=None):
        """
        Train all models and evaluate performance
        """
        print("Starting training of all models...")
        
        results = {}
        
        # Train deep learning models
        if len(X_train.shape) == 3:  # Sequential data
            # CNN-LSTM model
            cnn_lstm_model = self.create_cnn_lstm_model(X_train.shape[1:])
            self.train_deep_learning_model(cnn_lstm_model, X_train, y_train, X_val, y_val, 'cnn_lstm')
            results['cnn_lstm'] = self.evaluate_model(cnn_lstm_model, X_test, y_test, 'cnn_lstm')
            
            # LSTM model
            lstm_model = self.create_lstm_model(X_train.shape[1:])
            self.train_deep_learning_model(lstm_model, X_train, y_train, X_val, y_val, 'lstm')
            results['lstm'] = self.evaluate_model(lstm_model, X_test, y_test, 'lstm')
            
            # CNN-LSTM with satellite data (if available)
            if satellite_data is not None:
                # Create satellite input (simplified - using satellite features as additional input)
                satellite_features = np.mean(satellite_data, axis=(0, 1))  # Global average
                satellite_input = np.tile(satellite_features, (X_train.shape[0], X_train.shape[1], 1))
                
                cnn_lstm_sat_model = self.create_cnn_lstm_satellite_model(
                    X_train.shape[1:], 
                    satellite_input.shape[1:]
                )
                self.train_deep_learning_model(cnn_lstm_sat_model, X_train, y_train, X_val, y_val, 'cnn_lstm_satellite')
                results['cnn_lstm_satellite'] = self.evaluate_model(cnn_lstm_sat_model, X_test, y_test, 'cnn_lstm_satellite')
        
        # Train traditional models
        self.train_traditional_models(X_train, y_train, X_val, y_val)
        
        for name in ['random_forest', 'gradient_boosting', 'linear_regression', 'svr']:
            results[name] = self.evaluate_model(self.models[name], X_test, y_test, name)
        
        # Save all models
        for name in self.models.keys():
            self.save_model(self.models[name], name)
        
        return results

if __name__ == "__main__":
    # Test the models
    models = AQIPredictionModels()
    print("AQI prediction models initialized successfully!") 