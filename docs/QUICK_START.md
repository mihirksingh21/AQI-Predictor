# Quick Start Guide - AQI Prediction System

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
# Option A: Automatic installation (recommended)
python install_packages.py

# Option B: Manual installation
pip install -r minimal_requirements.txt
```

### Step 2: Test the System
```bash
# Test with simple demo (works with minimal dependencies)
python simple_demo.py

# Test with full demo (traditional ML models)
python main_no_tensorflow.py
# Choose option 2 for demo mode
```

### Step 3: Run with Real Data
```bash
# Run the full system
python main_no_tensorflow.py
# Choose option 1 and enter your city name
```

## 📊 What You'll Get

After running the system, you'll find:

### 📁 Generated Folders:
- `data/` - Collected air quality and weather data
- `models/` - Trained machine learning models
- `results/` - Visualizations and analysis
- `images/` - Satellite imagery (if available)

### 📈 Generated Files:
- `results/aqi_trends.png` - AQI trends over time
- `results/pollutant_correlations.png` - Correlation matrix
- `results/weather_aqi_relationship.png` - Weather vs AQI
- `results/model_comparison.png` - Model performance comparison
- `results/dashboard.html` - Interactive dashboard
- `results/aqi_map.html` - Interactive map

## 🔧 API Setup (Optional)

For better results, get free API keys:

### OpenWeatherMap API (Weather Data)
1. Go to [OpenWeatherMap API](https://openweathermap.org/api)
2. Sign up for free account
3. Get API key
4. Update `config.py`:
```python
OPENWEATHER_API_KEY = "your_api_key_here"
```

### NASA Earthdata (Satellite Data)
1. Register at [NASA Earthdata](https://urs.earthdata.nasa.gov/)
2. Update `config.py`:
```python
NASA_USERNAME = "your_username"
NASA_PASSWORD = "your_password"
```

**Note:** OpenAQ API is completely free and doesn't need any setup!

## 🎯 Example Usage

### Basic Prediction
```python
from main_no_tensorflow import AQIPredictionSystemNoTF

system = AQIPredictionSystemNoTF()
results = system.run_complete_pipeline(city="Delhi", days=30)
```

### Data Collection Only
```python
from data_collector import DataCollector

collector = DataCollector()
data = collector.collect_all_data(city="Mumbai", days=7)
print(f"Collected {len(data['aqi_data'])} air quality records")
```

## 🔍 Troubleshooting

### Common Issues:

1. **"No module named 'sklearn'"**
   ```bash
   pip install scikit-learn
   ```

2. **"No module named 'matplotlib'"**
   ```bash
   pip install matplotlib
   ```

3. **"No data found for city"**
   - Try different city names
   - Use demo mode for testing
   - Check if city has OpenAQ stations

4. **Poor model performance**
   - Use real data instead of demo data
   - Increase historical days in config
   - Check data quality

### Performance Tips:

- **Faster training**: Reduce `EPOCHS` in config.py
- **Better accuracy**: Increase `HISTORICAL_DAYS`
- **Memory optimization**: Reduce `BATCH_SIZE`

## 📚 Next Steps

1. **Read the full README.md** for detailed documentation
2. **Explore the code** in individual modules
3. **Customize the system** by editing config.py
4. **Add your own models** in models.py
5. **Create custom visualizations** in visualization.py

## 🆘 Need Help?

1. Check the troubleshooting section
2. Review the code comments
3. Run the simple demo first
4. Use demo mode for testing

---

**Remember:** This system uses only free APIs and datasets. No payment required! 