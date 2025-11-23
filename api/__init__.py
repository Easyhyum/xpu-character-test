"""
API module for XPU character test project.

This module provides model loading and text generation utilities.
"""

from .model_loader import model_load_function
from .text_generator import generate_with_activations, create_activation_hook
from .batch_generator import process_batch_inference
from .layer_operation_tracker import LayerOperationTracker

# TPU versions
from .model_loader_tpu import model_load_function_tpu
from .text_generator_tpu import generate_with_activations_tpu
from .batch_generator_tpu import process_batch_inference_tpu
from .layer_operation_tracker_tpu import LayerOperationTrackerTPU

__all__ = [
    'model_load_function',
    'generate_with_activations', 
    'create_activation_hook',
    'process_batch_inference',
    'LayerOperationTracker',
    # TPU versions
    'model_load_function_tpu',
    'generate_with_activations_tpu',
    'process_batch_inference_tpu',
    'LayerOperationTrackerTPU'
]