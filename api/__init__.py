"""
API module for XPU character test project.

This module provides model loading and text generation utilities.
"""

from .model_loader import model_load_function
from .text_generator import generate_with_activations, create_activation_hook

__all__ = [
    'model_load_function',
    'generate_with_activations', 
    'create_activation_hook'
]