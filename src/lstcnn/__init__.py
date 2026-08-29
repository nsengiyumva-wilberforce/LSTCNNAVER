"""Lightweight Spatio-Temporal CNN for audio-visual emotion recognition.

Faithful reimplementation of Ding, Tang, Lu, IEEE Trans. Affective Computing,
2025 (DOI: 10.1109/TAFFC.2025.3566773).
"""

from lstcnn.config import load_config
from lstcnn.flops import count_macs, gflops_from_macs
from lstcnn.model import LightweightSTCNN, count_parameters

__all__ = ["LightweightSTCNN", "count_parameters", "load_config"]
__version__ = "0.1.0"
