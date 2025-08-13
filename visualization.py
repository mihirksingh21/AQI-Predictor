"""
Visualization Module for AQI Prediction System
Creates charts, plots, and interactive visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import folium
from folium import plugins
import os
from config import *

class AQIVisualizer:
    def __init__(self):
        self.setup_style()
        
    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_aqi_trends(self, data, save_path=None):
        """
        Plot AQI trends over time
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AQI Trends and Analysis', fontsize=16)
        
        # AQI over time
        axes[0, 0].plot(data.index, data['aqi'], linewidth=2, color='red')
        axes[0, 0].set_title('AQI Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('AQI')
        axes[0, 0].grid(True, alpha=0.3)
        
        # AQI distribution
        axes[0, 1].hist(data['aqi'], bins=30, alpha=0.7, color='orange')
        axes[0, 1].set_title('AQI Distribution')
        axes[0, 1].set_xlabel('AQI')
        axes[0, 1].set_ylabel('Frequency')
        
        # AQI by hour
        hourly_aqi = data.groupby(data.index.hour)['aqi'].mean()
        axes[1, 0].bar(hourly_aqi.index, hourly_aqi.values, alpha=0.7, color='green')
        axes[1, 0].set_title('Average AQI by Hour')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average AQI')
        
        # AQI by day of week
        daily_aqi = data.groupby(data.index.dayofweek)['aqi'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(days, daily_aqi.values, alpha=0.7, color='purple')
        axes[1, 1].set_title('Average AQI by Day of Week')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Average AQI')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pollutant_correlations(self, data, save_path=None):
        """
        Plot correlation matrix of pollutants
        """
        pollutant_cols = [col for col in POLLUTANTS if col in data.columns]
        
        if len(pollutant_cols) < 2:
            print("Not enough pollutant data for correlation plot")
            return
        
        # Calculate correlation matrix
        corr_matrix = data[pollutant_cols].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Pollutant Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_weather_aqi_relationship(self, data, save_path=None):
        """
        Plot relationship between weather variables and AQI
        """
        weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        available_cols = [col for col in weather_cols if col in data.columns]
        
        if len(available_cols) < 2:
            print("Not enough weather data for relationship plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Weather-AQI Relationships', fontsize=16)
        
        for i, col in enumerate(available_cols[:4]):
            row = i // 2
            col_idx = i % 2
            
            axes[row, col_idx].scatter(data[col], data['aqi'], alpha=0.6, s=20)
            axes[row, col_idx].set_xlabel(col.title())
            axes[row, col_idx].set_ylabel('AQI')
            axes[row, col_idx].set_title(f'{col.title()} vs AQI')
            axes[row, col_idx].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(data[col], data['aqi'], 1)
            p = np.poly1d(z)
            axes[row, col_idx].plot(data[col], p(data[col]), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_predictions(self, y_true, y_pred, model_name, save_path=None):
        """
        Plot model predictions vs actual values
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{model_name} Predictions vs Actual', fontsize=16)
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=30)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual AQI')
        axes[0].set_ylabel('Predicted AQI')
        axes[0].set_title('Predictions vs Actual')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=30)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted AQI')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, results, save_path=None):
        """
        Compare performance of different models
        """
        models = list(results.keys())
        metrics = ['rmse', 'mae', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            values = [results[model][0][metric] for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_aqi_map(self, data, save_path=None):
        """
        Create interactive map showing AQI data
        """
        if 'coordinates.latitude' not in data.columns or 'coordinates.longitude' not in data.columns:
            print("No location data available for map")
            return
        
        # Create base map
        center_lat = data['coordinates.latitude'].mean()
        center_lon = data['coordinates.longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add AQI markers
        for idx, row in data.iterrows():
            if pd.notna(row['coordinates.latitude']) and pd.notna(row['coordinates.longitude']):
                aqi_value = row.get('aqi', 50)
                aqi_category = self.get_aqi_category(aqi_value)
                
                # Color based on AQI category
                color_map = {
                    'Good': 'green',
                    'Moderate': 'yellow',
                    'Unhealthy for Sensitive Groups': 'orange',
                    'Unhealthy': 'red',
                    'Very Unhealthy': 'purple',
                    'Hazardous': 'maroon'
                }
                
                color = color_map.get(aqi_category, 'gray')
                
                folium.CircleMarker(
                    location=[row['coordinates.latitude'], row['coordinates.longitude']],
                    radius=8,
                    popup=f"AQI: {aqi_value:.1f}<br>Category: {aqi_category}",
                    color=color,
                    fill=True,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>AQI Categories</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Good (0-50)</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Moderate (51-100)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Unhealthy for Sensitive Groups (101-150)</p>
        <p><i class="fa fa-circle" style="color:red"></i> Unhealthy (151-200)</p>
        <p><i class="fa fa-circle" style="color:purple"></i> Very Unhealthy (201-300)</p>
        <p><i class="fa fa-circle" style="color:maroon"></i> Hazardous (301-500)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(save_path)
        
        return m
    
    def create_interactive_dashboard(self, data, predictions=None, save_path=None):
        """
        Create interactive dashboard using Plotly
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('AQI Over Time', 'AQI Distribution', 'Weather vs AQI', 
                          'Hourly AQI Pattern', 'Daily AQI Pattern', 'Model Predictions'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # AQI over time
        fig.add_trace(
            go.Scatter(x=data.index, y=data['aqi'], mode='lines', name='AQI',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # AQI distribution
        fig.add_trace(
            go.Histogram(x=data['aqi'], nbinsx=30, name='AQI Distribution',
                        marker_color='orange', opacity=0.7),
            row=1, col=2
        )
        
        # Weather vs AQI (temperature)
        if 'temperature' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['temperature'], y=data['aqi'], mode='markers',
                          name='Temperature vs AQI', marker=dict(size=5, opacity=0.6)),
                row=2, col=1
            )
        
        # Hourly pattern
        hourly_aqi = data.groupby(data.index.hour)['aqi'].mean()
        fig.add_trace(
            go.Bar(x=hourly_aqi.index, y=hourly_aqi.values, name='Hourly AQI',
                  marker_color='green', opacity=0.7),
            row=2, col=2
        )
        
        # Daily pattern
        daily_aqi = data.groupby(data.index.dayofweek)['aqi'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig.add_trace(
            go.Bar(x=days, y=daily_aqi.values, name='Daily AQI',
                  marker_color='purple', opacity=0.7),
            row=3, col=1
        )
        
        # Model predictions (if available)
        if predictions is not None:
            fig.add_trace(
                go.Scatter(x=data.index[-len(predictions):], y=predictions, 
                          mode='lines', name='Predictions',
                          line=dict(color='blue', width=2, dash='dash')),
                row=3, col=2
            )
        
        fig.update_layout(height=900, title_text="AQI Prediction Dashboard")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def get_aqi_category(self, aqi_value):
        """Get AQI category based on value"""
        for (low, high), category in AQI_CATEGORIES.items():
            if low <= aqi_value <= high:
                return category
        return "Unknown"
    
    def create_all_visualizations(self, data, results=None, save_dir=RESULTS_DIR):
        """
        Create all visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating visualizations...")
        
        # Basic plots
        self.plot_aqi_trends(data, os.path.join(save_dir, 'aqi_trends.png'))
        self.plot_pollutant_correlations(data, os.path.join(save_dir, 'pollutant_correlations.png'))
        self.plot_weather_aqi_relationship(data, os.path.join(save_dir, 'weather_aqi_relationship.png'))
        
        # Interactive map
        map_obj = self.create_interactive_aqi_map(data, os.path.join(save_dir, 'aqi_map.html'))
        
        # Interactive dashboard
        dashboard = self.create_interactive_dashboard(data, save_path=os.path.join(save_dir, 'dashboard.html'))
        
        # Model comparison (if results available)
        if results:
            self.plot_model_comparison(results, os.path.join(save_dir, 'model_comparison.png'))
        
        print(f"All visualizations saved to {save_dir}")

if __name__ == "__main__":
    # Test the visualizer
    visualizer = AQIVisualizer()
    print("AQI visualizer initialized successfully!") 