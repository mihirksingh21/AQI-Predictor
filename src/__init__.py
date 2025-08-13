"""
AQI Prediction System - Main Package

This package contains the core components for air quality prediction:
- Core system functionality
- Machine learning models
- Data collection and preprocessing
- Visualization tools
- Utility functions
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Advanced Air Quality Index Prediction System using Machine Learning and Satellite Data"

# Import main components for easy access
from .core import AQIPredictionSystem, AQIPredictionSystemNoTF
from .models import *
from .data import *
from .visualization import *

__all__ = [
    'AQIPredictionSystem',
    'AQIPredictionSystemNoTF',
] 