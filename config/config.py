# Configuration file for AQI Prediction System
# All APIs and datasets used are FREE

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (FREE APIs - No payment required)
# OpenWeatherMap API - Free tier: 1000 calls/day
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")

# NASA Earthdata - Completely free, just need to register
NASA_USERNAME = os.getenv("NASA_USERNAME", "YOUR_NASA_USERNAME")
NASA_PASSWORD = os.getenv("NASA_PASSWORD", "YOUR_NASA_PASSWORD")

# OpenAQ API - Completely free, no API key required
OPENAQ_BASE_URL = os.getenv("OPENAQ_BASE_URL", "https://api.openaq.org/v2")

# Default city for predictions
DEFAULT_CITY = "Delhi"
DEFAULT_COUNTRY = "IN"

# Satellite data settings
SATELLITE_IMAGE_SIZE = (224, 224)
SATELLITE_BANDS = ['B2', 'B3', 'B4', 'B8']  # Blue, Green, Red, NIR for Sentinel-2

# Model parameters
MODEL_INPUT_SIZE = (224, 224, 4)  # Height, Width, Channels
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# Data collection settings
HISTORICAL_DAYS = 30  # Days of historical data to collect
PREDICTION_HORIZON = 24  # Hours ahead to predict

# File paths
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
IMAGES_DIR = "images"

# AQI Categories
AQI_CATEGORIES = {
    (0, 50): "Good",
    (51, 100): "Moderate", 
    (101, 150): "Unhealthy for Sensitive Groups",
    (151, 200): "Unhealthy",
    (201, 300): "Very Unhealthy",
    (301, 500): "Hazardous"
}

# Air pollutants to monitor
POLLUTANTS = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3'] 