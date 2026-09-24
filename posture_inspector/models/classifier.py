import os
import joblib
import numpy as np
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

CLASSES = ["sitting", "standing", "bending", "slouching"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

class PosturePyTorchNet(nn.Module if HAS_TORCH else object):
    """
    Deep Neural Network (MLP) for Posture Classification using PyTorch.
    Input vector size: 108 features.
    Output: 4 posture classes ('sitting', 'standing', 'bending', 'slouching').
    """
    def __init__(self, input_dim=108, num_classes=len(CLASSES)):
        super(PosturePyTorchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class PostureClassifierManager:
    """
    Unified Inference Manager for PyTorch & Scikit-Learn Posture Classification Models.
    """
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir
        self.pytorch_model_path = os.path.join(model_dir, "posture_net.pth")
        self.sklearn_model_path = os.path.join(model_dir, "posture_rf.pkl")
        
        self.pytorch_model = None
        self.sklearn_model = None
        self._load_models()

    def _load_models(self):
        # Load PyTorch Model if available
        if HAS_TORCH and os.path.exists(self.pytorch_model_path):
            try:
                model = PosturePyTorchNet()
                model.load_state_dict(torch.load(self.pytorch_model_path, map_location=torch.device('cpu'), weights_only=True))
                model.eval()
                self.pytorch_model = model
            except Exception as e:
                print(f"[Warning] Failed to load PyTorch model weights: {e}")

        # Load Scikit-Learn Model if available
        if os.path.exists(self.sklearn_model_path):
            try:
                self.sklearn_model = joblib.load(self.sklearn_model_path)
            except Exception as e:
                print(f"[Warning] Failed to load Scikit-Learn model: {e}")

    def predict(self, feature_vector, use_pytorch=True):
        """
        Predict posture label & confidence from feature vector.
        Returns: (label, confidence_score)
        """
        feature_vector = np.array(feature_vector, dtype=np.float32)
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        if use_pytorch and self.pytorch_model is not None:
            with torch.no_grad():
                tensor_input = torch.from_numpy(feature_vector)
                logits = self.pytorch_model(tensor_input)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                idx = int(np.argmax(probs))
                return IDX_TO_CLASS.get(idx, "unknown"), float(probs[idx])

        elif self.sklearn_model is not None:
            probs = self.sklearn_model.predict_proba(feature_vector)[0]
            idx = int(np.argmax(probs))
            classes = getattr(self.sklearn_model, "classes_", CLASSES)
            label = classes[idx] if idx < len(classes) else IDX_TO_CLASS.get(idx, "unknown")
            return str(label), float(probs[idx])

        # Rule-based fallback if models are not pre-built yet
        return self._heuristic_fallback(feature_vector[0])

    def _heuristic_fallback(self, vec):
        """
        Heuristic fallback classifier using calculated biomechanical angles.
        Metric values are at the end of feature vector:
        vec[-9]: torso_spine_angle / 90
        vec[-8]: left_knee_angle / 180
        vec[-7]: right_knee_angle / 180
        vec[-6]: avg_knee_angle / 180
        vec[-5]: left_hip_angle / 180
        vec[-4]: right_hip_angle / 180
        vec[-3]: avg_hip_angle / 180
        vec[-2]: neck_inclination / 90
        """
        torso_angle = vec[-9] * 90.0
        avg_knee = vec[-6] * 180.0
        avg_hip = vec[-3] * 180.0
        neck_inc = vec[-2] * 90.0

        if torso_angle > 35.0:
            return "bending", 0.92
        elif avg_knee < 120.0 and avg_hip < 130.0:
            return "sitting", 0.95
        elif neck_inc > 25.0:
            return "slouching", 0.88
        else:
            return "standing", 0.90

# Alias for backward compatibility
MLPostureClassifier = PostureClassifierManager
