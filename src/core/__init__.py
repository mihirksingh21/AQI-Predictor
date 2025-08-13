"""
Core system functionality for AQI Prediction System

This module contains the main system classes and entry points.
"""

from .main import AQIPredictionSystem
from .main_no_tensorflow import AQIPredictionSystemNoTF

__all__ = [
    'AQIPredictionSystem',
    'AQIPredictionSystemNoTF',
] 