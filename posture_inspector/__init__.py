"""
PostureInspector: Industrial Computer Vision & Edge AI Posture Analysis Engine.
"""

from .tracker import PoseTracker
from .feature_extractor import FeatureExtractor
from .models.classifier import PosturePyTorchNet, MLPostureClassifier
from .edge_integration.plc_dispatcher import IndustrialPLCDispatcher

__version__ = "2.0.0"
__all__ = [
    "PoseTracker",
    "FeatureExtractor",
    "PosturePyTorchNet",
    "MLPostureClassifier",
    "IndustrialPLCDispatcher",
]
