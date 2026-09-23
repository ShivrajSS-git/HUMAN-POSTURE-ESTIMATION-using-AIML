import sys
sys.modules['tensorflow'] = None

import pytest
import numpy as np
import cv2

from posture_inspector.feature_extractor import FeatureExtractor
from posture_inspector.models.classifier import PostureClassifierManager
from posture_inspector.edge_integration.plc_dispatcher import IndustrialPLCDispatcher
from posture_inspector.tracker import PoseTracker

def test_feature_extractor_angles():
    # Angle between (0, 1), (0, 0), (1, 0) should be 90 degrees
    a = (0.0, 1.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (1.0, 0.0, 0.0)
    angle = FeatureExtractor.calculate_angle(a, b, c)
    assert abs(angle - 90.0) < 1e-3

def test_feature_vector_length():
    class LandmarkMock:
        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    mock_landmarks = [LandmarkMock(0.1*i, 0.05*i, 0.01*i) for i in range(33)]
    vec = FeatureExtractor.extract_feature_vector(mock_landmarks)
    assert len(vec) == 108

def test_classifier_manager():
    clf = PostureClassifierManager()
    dummy_feat = np.random.randn(108)
    label, conf = clf.predict(dummy_feat)
    assert label in ["sitting", "standing", "bending", "slouching", "unknown"]
    assert 0.0 <= conf <= 1.0

def test_plc_dispatcher():
    plc = IndustrialPLCDispatcher()
    res = plc.dispatch_signal("slouching", 0.92, "HIGH RISK", 12.5)
    assert res["plc_digital_outputs"]["PLC_OUT_ALARM_LIGHT"] == 1
    assert res["plc_digital_outputs"]["PLC_OUT_NORMAL_OP"] == 0

def test_pose_tracker_synthetic_frame():
    tracker = PoseTracker(static_image_mode=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    processed_frame, info = tracker.process_frame(frame, draw_hud=True)
    assert processed_frame.shape == (480, 640, 3)
    assert "pose_detected" in info
    tracker.close()
