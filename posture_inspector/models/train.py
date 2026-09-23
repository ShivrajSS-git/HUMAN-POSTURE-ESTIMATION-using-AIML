import sys
sys.modules['tensorflow'] = None

import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from posture_inspector.feature_extractor import FeatureExtractor
from posture_inspector.models.classifier import (
    CLASSES,
    CLASS_TO_IDX,
    PosturePyTorchNet,
)

class SyntheticLandmarkGenerator:
    """
    Generates realistic biomechanical posture feature vectors for dataset augmentation and model training.
    """
    @staticmethod
    def generate_samples_for_class(cls_name, num_samples=300):
        np.random.seed(42 if cls_name == 'sitting' else 123)
        features_list = []

        for _ in range(num_samples):
            pts = np.zeros((33, 3))
            
            if cls_name == "sitting":
                torso_spine = np.random.uniform(0, 15)
                knee_angle = np.random.uniform(80, 110)
                hip_angle = np.random.uniform(80, 110)
                neck_inc = np.random.uniform(5, 20)
            elif cls_name == "standing":
                torso_spine = np.random.uniform(0, 10)
                knee_angle = np.random.uniform(165, 180)
                hip_angle = np.random.uniform(165, 180)
                neck_inc = np.random.uniform(2, 15)
            elif cls_name == "bending":
                torso_spine = np.random.uniform(35, 75)
                knee_angle = np.random.uniform(130, 175)
                hip_angle = np.random.uniform(45, 90)
                neck_inc = np.random.uniform(15, 35)
            elif cls_name == "slouching":
                torso_spine = np.random.uniform(15, 30)
                knee_angle = np.random.uniform(80, 110)
                hip_angle = np.random.uniform(70, 100)
                neck_inc = np.random.uniform(30, 55)
            else:
                torso_spine = 10
                knee_angle = 170
                hip_angle = 170
                neck_inc = 10

            noise = np.random.normal(0, 0.02, (33, 3))
            
            pts[11] = [-0.2, 0.4, 0.0]
            pts[12] = [0.2, 0.4, 0.0]
            pts[23] = [-0.15, 0.0, 0.0]
            pts[24] = [0.15, 0.0, 0.0]
            pts[25] = [-0.15, -0.4, 0.0]
            pts[26] = [0.15, -0.4, 0.0]
            pts[27] = [-0.15, -0.8, 0.0]
            pts[28] = [0.15, -0.8, 0.0]
            pts[7] = [-0.1, 0.6, 0.0]
            pts[8] = [0.1, 0.6, 0.0]

            pts += noise

            class LandmarkMock:
                def __init__(self, x, y, z):
                    self.x = float(x)
                    self.y = float(y)
                    self.z = float(z)

            lm_objs = [LandmarkMock(p[0], p[1], p[2]) for p in pts]
            
            vec = FeatureExtractor.extract_feature_vector(lm_objs)
            
            vec[-9] = torso_spine / 90.0
            vec[-6] = knee_angle / 180.0
            vec[-3] = hip_angle / 180.0
            vec[-2] = neck_inc / 90.0

            features_list.append(vec)

        return np.array(features_list)


def train_models():
    """
    Main training pipeline for PyTorch MLP and Scikit-Learn Random Forest.
    """
    print("==================================================")
    print(" [*] Starting Posture Inspector Model Training Pipeline")
    print("==================================================")

    X_list = []
    y_list = []

    # 1. Load real images dataset if available
    dataset_dir = "HUMAN POSTURE ESTIMATION/images_dataset"
    if os.path.exists(dataset_dir):
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True) as pose:
            for label in os.listdir(dataset_dir):
                folder = os.path.join(dataset_dir, label)
                if not os.path.isdir(folder):
                    continue
                norm_label = label.lower().strip()
                if norm_label not in CLASS_TO_IDX:
                    continue

                for img_file in os.listdir(folder):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(folder, img_file)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        if results.pose_landmarks:
                            vec = FeatureExtractor.extract_feature_vector(results.pose_landmarks.landmark)
                            X_list.append(vec)
                            y_list.append(CLASS_TO_IDX[norm_label])

    print(f"[*] Real dataset samples extracted: {len(X_list)}")

    # 2. Augment dataset with synthetic samples for robust multi-class coverage
    synth_gen = SyntheticLandmarkGenerator()
    for cls_name in CLASSES:
        synth_X = synth_gen.generate_samples_for_class(cls_name, num_samples=250)
        synth_y = np.full(len(synth_X), CLASS_TO_IDX[cls_name])
        X_list.extend(synth_X)
        y_list.extend(synth_y)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"[*] Total combined dataset size: {X.shape[0]} samples, {X.shape[1]} features.")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --------------------------------------------------
    # 3. Train Scikit-Learn Random Forest Classifier
    # --------------------------------------------------
    print("\n[*] Training Scikit-Learn Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    rf_preds = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"[OK] Random Forest Accuracy: {rf_acc * 100:.2f}%")

    model_dir = os.path.dirname(os.path.abspath(__file__))
    rf_path = os.path.join(model_dir, "posture_rf.pkl")
    joblib.dump(rf_clf, rf_path)
    print(f"[SAVE] Saved Random Forest model to: {rf_path}")

    # --------------------------------------------------
    # 4. Train PyTorch Deep Neural Network (MLP)
    # --------------------------------------------------
    print("\n[*] Training PyTorch Neural Network (MLP)...")
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    net = PosturePyTorchNet(input_dim=X.shape[1], num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-4)

    epochs = 40
    net.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    net.eval()
    with torch.no_grad():
        test_logits = net(torch.from_numpy(X_test))
        torch_preds = torch.argmax(test_logits, dim=1).numpy()
        torch_acc = accuracy_score(y_test, torch_preds)

    print(f"[OK] PyTorch Model Accuracy: {torch_acc * 100:.2f}%")
    print("\n[*] Detailed Classification Report:")
    print(classification_report(y_test, torch_preds, target_names=CLASSES))

    torch_path = os.path.join(model_dir, "posture_net.pth")
    torch.save(net.state_dict(), torch_path)
    print(f"[SAVE] Saved PyTorch model to: {torch_path}")

    # --------------------------------------------------
    # 5. Export to ONNX Model for Edge AI Deployment
    # --------------------------------------------------
    try:
        onnx_path = os.path.join(model_dir, "posture_net.onnx")
        dummy_input = torch.randn(1, X.shape[1], dtype=torch.float32)
        torch.onnx.export(
            net,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_feature_vector'],
            output_names=['output_logits'],
            dynamic_axes={'input_feature_vector': {0: 'batch_size'}, 'output_logits': {0: 'batch_size'}}
        )
        print(f"[SAVE] Exported ONNX model for Edge Devices to: {onnx_path}")
    except Exception as e:
        print(f"[!] ONNX Export Notice: {e}")

    print("\n[SUCCESS] Training Pipeline Completed Successfully!")

if __name__ == "__main__":
    train_models()
