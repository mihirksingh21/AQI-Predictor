"""
Data Collector Module for AQI Prediction System
Collects data from free APIs: OpenAQ, OpenWeatherMap, and NASA Earthdata
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time
from PIL import Image
import rasterio
from rasterio.plot import show
import earthaccess
from config.config import *

class DataCollector:
    def __init__(self):
        self.setup_directories()
        self.setup_apis()
    
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, IMAGES_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_apis(self):
        """Setup API connections"""
        # OpenAQ doesn't require API key
        self.openaq_url = OPENAQ_BASE_URL
        
        # OpenWeatherMap API
        self.weather_api_key = OPENWEATHER_API_KEY
        
        # NASA Earthdata setup
        try:
            earthaccess.login(strategy="netrc")
            print("NASA Earthdata login successful")
        except:
            print("NASA Earthdata login failed. Please register at https://urs.earthdata.nasa.gov/")
    
    def get_air_quality_data(self, city=DEFAULT_CITY, country=DEFAULT_COUNTRY, days=HISTORICAL_DAYS):
        """
        Collect air quality data from OpenAQ API (FREE)
        """
        print(f"Collecting air quality data for {city}, {country}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # OpenAQ API parameters
        params = {
            'city': city,
            'country': country,
            'limit': 10000,  # Maximum allowed
            'date_from': start_date.strftime('%Y-%m-%d'),
            'date_to': end_date.strftime('%Y-%m-%d'),
            'order_by': 'datetime'
        }
        
        try:
            response = requests.get(f"{self.openaq_url}/measurements", params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['results']:
                # Convert to DataFrame
                df = pd.json_normalize(data['results'])
                
                # Clean and process data
                df['datetime'] = pd.to_datetime(df['date.utc'])
                df['date'] = df['datetime'].dt.date
                df['hour'] = df['datetime'].dt.hour
                
                # Select relevant columns
                columns_to_keep = ['datetime', 'date', 'hour', 'parameter', 'value', 'unit', 
                                 'location', 'coordinates.latitude', 'coordinates.longitude']
                df = df[columns_to_keep]
                
                # Pivot to get pollutants as columns
                df_pivot = df.pivot_table(
                    index=['datetime', 'date', 'hour', 'location', 'coordinates.latitude', 'coordinates.longitude'],
                    columns='parameter',
                    values='value',
                    aggfunc='mean'
                ).reset_index()
                
                # Save raw data
                df_pivot.to_csv(f"{DATA_DIR}/{city}_air_quality_raw.csv", index=False)
                print(f"Collected {len(df_pivot)} air quality records")
                
                return df_pivot
            else:
                print("No air quality data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error collecting air quality data: {e}")
            return pd.DataFrame()
    
    def get_weather_data(self, city=DEFAULT_CITY, days=HISTORICAL_DAYS):
        """
        Collect weather data from OpenWeatherMap API (FREE - 1000 calls/day)
        """
        print(f"Collecting weather data for {city}")
        
        if self.weather_api_key == "YOUR_OPENWEATHER_API_KEY":
            print("Please set your OpenWeatherMap API key in config.py")
            print("Get free API key from: https://openweathermap.org/api")
            return pd.DataFrame()
        
        # Get city coordinates first
        geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct"
        geo_params = {
            'q': f"{city},{DEFAULT_COUNTRY}",
            'limit': 1,
            'appid': self.weather_api_key
        }
        
        try:
            geo_response = requests.get(geocoding_url, params=geo_params)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            
            if not geo_data:
                print(f"City {city} not found")
                return pd.DataFrame()
            
            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
            
            # Get historical weather data
            weather_data = []
            current_date = datetime.now() - timedelta(days=days)
            
            for i in range(days):
                date = current_date + timedelta(days=i)
                timestamp = int(date.timestamp())
                
                weather_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
                weather_params = {
                    'lat': lat,
                    'lon': lon,
                    'dt': timestamp,
                    'appid': self.weather_api_key,
                    'units': 'metric'
                }
                
                response = requests.get(weather_url, params=weather_params)
                if response.status_code == 200:
                    data = response.json()
                    for hour_data in data.get('hourly', []):
                        weather_data.append({
                            'datetime': datetime.fromtimestamp(hour_data['dt']),
                            'temperature': hour_data['temp'],
                            'humidity': hour_data['humidity'],
                            'pressure': hour_data['pressure'],
                            'wind_speed': hour_data['wind_speed'],
                            'wind_deg': hour_data['wind_deg'],
                            'clouds': hour_data['clouds'],
                            'weather_main': hour_data['weather'][0]['main'],
                            'weather_description': hour_data['weather'][0]['description']
                        })
                
                time.sleep(0.1)  # Rate limiting
            
            if weather_data:
                df = pd.DataFrame(weather_data)
                df.to_csv(f"{DATA_DIR}/{city}_weather_data.csv", index=False)
                print(f"Collected {len(df)} weather records")
                return df
            else:
                print("No weather data collected")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error collecting weather data: {e}")
            return pd.DataFrame()
    
    def get_satellite_data(self, city=DEFAULT_CITY, days=7):
        """
        Collect satellite imagery from NASA Earthdata (FREE)
        """
        print(f"Collecting satellite data for {city}")
        
        try:
            # Search for Sentinel-2 data
            dataset = earthaccess.search_data(
                short_name="S2MSI2A",  # Sentinel-2 Level-2A
                temporal=(datetime.now() - timedelta(days=days), datetime.now()),
                spatial=(-180, -90, 180, 90)  # Global search, will filter by city later
            )
            
            if dataset:
                print(f"Found {len(dataset)} satellite images")
                
                # Download the most recent image
                files = earthaccess.download(dataset[:1])
                
                if files:
                    print(f"Downloaded satellite image: {files[0]}")
                    return files[0]
                else:
                    print("Failed to download satellite image")
                    return None
            else:
                print("No satellite data found")
                return None
                
        except Exception as e:
            print(f"Error collecting satellite data: {e}")
            return None
    
    def process_satellite_image(self, image_path):
        """
        Process satellite image for model input
        """
        if not image_path or not os.path.exists(image_path):
            print("No satellite image to process")
            return None
        
        try:
            with rasterio.open(image_path) as src:
                # Read RGB bands (B2, B3, B4) and NIR (B8)
                bands = []
                for band_name in SATELLITE_BANDS:
                    band_idx = int(band_name[1])  # Extract band number
                    band_data = src.read(band_idx)
                    bands.append(band_data)
                
                # Stack bands
                image_array = np.stack(bands, axis=-1)
                
                # Resize to model input size
                from skimage.transform import resize
                image_resized = resize(image_array, SATELLITE_IMAGE_SIZE + (len(bands),))
                
                # Normalize to 0-1
                image_normalized = image_resized / np.max(image_resized)
                
                # Save processed image
                processed_path = f"{IMAGES_DIR}/processed_satellite_image.npy"
                np.save(processed_path, image_normalized)
                
                print(f"Processed satellite image saved to {processed_path}")
                return image_normalized
                
        except Exception as e:
            print(f"Error processing satellite image: {e}")
            return None
    
    def collect_all_data(self, city=DEFAULT_CITY, days=HISTORICAL_DAYS):
        """
        Collect all types of data for the AQI prediction system
        """
        print(f"Starting data collection for {city}")
        
        # Collect air quality data
        aqi_data = self.get_air_quality_data(city, days=days)
        
        # Collect weather data
        weather_data = self.get_weather_data(city, days=days)
        
        # Collect satellite data
        satellite_file = self.get_satellite_data(city, days=7)
        satellite_data = self.process_satellite_image(satellite_file)
        
        # Merge data if possible
        if not aqi_data.empty and not weather_data.empty:
            # Merge on datetime
            merged_data = pd.merge(
                aqi_data, 
                weather_data, 
                on='datetime', 
                how='outer'
            )
            
            # Save merged data
            merged_data.to_csv(f"{DATA_DIR}/{city}_merged_data.csv", index=False)
            print(f"Merged data saved with {len(merged_data)} records")
        
        return {
            'aqi_data': aqi_data,
            'weather_data': weather_data,
            'satellite_data': satellite_data
        }

if __name__ == "__main__":
    collector = DataCollector()
    data = collector.collect_all_data()
    print("Data collection completed!") 