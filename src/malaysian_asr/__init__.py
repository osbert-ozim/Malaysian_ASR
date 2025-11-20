"""
Malaysian ASR Package

A Python package for Malaysian Speech Recognition using MERaLiON-2-10B-ASR model.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.transcriber import MERaLiONTranscriber
from .core.batch_transcriber import BatchTranscriber
from .core.benchmark import RTFBenchmark
from .utils.accuracy import evaluate_accuracy
from .utils.model_downloader import download_model

__all__ = [
    "MERaLiONTranscriber",
    "BatchTranscriber", 
    "RTFBenchmark",
    "evaluate_accuracy",
    "download_model",
]
