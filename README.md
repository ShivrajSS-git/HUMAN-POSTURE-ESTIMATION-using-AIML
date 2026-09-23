# 🧍‍♂️ Industrial Human Posture AI Inspector & Edge Ergonomics Solution

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-teal.svg)](https://google.github.io/mediapipe/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Live%20Dashboard-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **Industrial Computer Vision & Edge AI Inspection System** for real-time human posture classification, ergonomic strain monitoring, and simulated **PLC (Programmable Logic Controller) digital I/O safety triggers**.

Designed for workplace safety, industrial quality assurance, and edge AI inspection pipelines.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Webcam / Video Stream / Camera Feed] --> B[OpenCV Frame Capture & Preprocessing]
    B --> C[MediaPipe 33 Landmark Pose Estimator]
    C --> D[Feature Engineering Engine\n33 3D Keypoints + Biomechanical Joint Angles]
    D --> E1[PyTorch Deep Neural Network MLP]
    D --> E2[Scikit-Learn Random Forest Classifier]
    E1 --> F[Post-Processing & Ergonomic Risk Engine]
    E2 --> F
    F --> G[Industrial PLC Interlock Simulator\nDigital I/O Pins & Relay Telemetry]
    F --> H[Live Web Inspection Dashboard\nFastAPI + MJPEG Video Stream + REST API]
```

---

## 🔥 Key Technical Features

1. **Real-Time Pose Tracking & Landmark Extraction**:
   - Uses **OpenCV** and **MediaPipe Pose** to detect and track 33 3D body keypoints in real time.
   - Computes scale-normalized spatial coordinates invariant to operator distance and camera zoom.

2. **Biomechanical Feature Engineering**:
   - Calculates real-time joint angles: Torso Spine Inclination, Neck Angle, Knee Flexion/Extension, Hip Angle, and Shoulder Tilt.
   - Generates a robust **108-dimensional feature vector** combining 99 normalized landmark coordinates + 9 biomechanical angle ratios.

3. **Multi-Model AI Classification Pipeline**:
   - **PyTorch Neural Network (MLP)**: Deep learning model built with BatchNorm, Dropout, and ReLU activations (**>99.0% accuracy**).
   - **Scikit-Learn Random Forest Classifier**: High-speed ML model for lightweight edge deployment (**99.5% accuracy**).
   - Automated dataset synthesis and noise augmentation pipeline for `sitting`, `standing`, `bending`, and `slouching` posture classes.

4. **Industrial PLC & Edge IoT Signal Dispatcher**:
   - Simulates hardware digital I/O relay pins for automated industrial safety interlocks:
     - `PLC_OUT_NORMAL_OP`: High signal (1) during healthy ergonomic operations.
     - `PLC_OUT_ALARM_LIGHT`: Triggered during posture strain or slouching.
     - `PLC_OUT_BUZZER_ALERT`: Audible warning signal for operator ergonomic risk.
     - `PLC_OUT_CONVEYOR_HALT`: Safety stop relay signal for hazardous bending postures.
   - Emits real-time JSON telemetry over REST API endpoints.

5. **Live Web Inspection Dashboard**:
   - Real-time **FastAPI** web application serving an **MJPEG live video stream** with augmented reality HUD overlays.
   - Live telemetry status gauges, FPS performance counters, and PLC relay state indicators.

---

## 📊 Model Performance & Benchmarks

| Model | Architecture | Accuracy | Inference Latency | Feature Input |
|---|---|---|---|---|
| **PyTorch MLP** | 4-layer Deep Neural Net | **99.00%** | ~12 ms | 108 Dims |
| **Random Forest** | 100 Trees Ensemble | **99.50%** | ~4 ms | 108 Dims |

---

## 🚀 Quickstart & Usage

### 1. Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/ShivrajSS-git/HUMAN-POSTURE-ESTIMATION-using-AIML.git
cd HUMAN-POSTURE-ESTIMATION-using-AIML
pip install -r requirements.txt
```

### 2. Run Test Image Inspection Mode

Inspect a sample image file with full HUD overlays and PLC telemetry output:

```bash
python main.py --mode test-image
```

### 3. Run Live Webcam Pose Inspector

Run real-time pose tracking and posture prediction on your local camera (Press `'q'` to exit, `'s'` to save snapshot):

```bash
python main.py --mode webcam
```

### 4. Launch Live Web Inspection Dashboard

Start the FastAPI web dashboard server and open `http://localhost:8000` in your browser:

```bash
python main.py --mode dashboard --port 8000
```

### 5. Train & Evaluate Models

Train PyTorch & Scikit-Learn models with dataset loading and data augmentation:

```bash
python main.py --mode train
```

### 6. Run Automated Test Suite

Run the `pytest` suite to verify feature extraction, model inference, and PLC dispatcher logic:

```bash
python -m pytest tests/
```

---

## 📁 Repository Structure

```
.
├── posture_inspector/
│   ├── tracker.py                 # OpenCV + MediaPipe pose tracker & HUD visualizer
│   ├── feature_extractor.py       # Joint angle & 108-dim landmark feature engine
│   ├── models/
│   │   ├── classifier.py          # PyTorch MLP & Scikit-Learn model wrappers
│   │   ├── train.py               # Dataset loader, training & evaluation pipeline
│   │   ├── posture_net.pth        # Trained PyTorch model weights
│   │   └── posture_rf.pkl         # Trained Random Forest model weights
│   ├── edge_integration/
│   │   └── plc_dispatcher.py      # Industrial PLC digital I/O & telemetry simulator
│   └── dashboard/
│       └── app.py                 # FastAPI live MJPEG video stream & web dashboard
├── HUMAN POSTURE ESTIMATION/
│   ├── code.py                    # Legacy entry script (upgraded)
│   ├── images_dataset/            # Image dataset directories
│   └── test_images/               # Test image samples
├── tests/
│   └── test_pipeline.py           # Pytest unit test suite
├── main.py                        # Unified CLI entry point
├── requirements.txt               # Dependencies list
└── README.md                      # System documentation
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
