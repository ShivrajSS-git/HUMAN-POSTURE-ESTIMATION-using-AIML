# 🧍‍♂️ Human Posture Estimation using AI/ML

This project classifies human postures (e.g., **sitting**, **standing**, **bending**) from **static images** using **MediaPipe Pose Estimation** and a **Machine Learning model** (Random Forest). It is built in **Python** and does not require a webcam.

---

## 📌 Features

- Pose landmark extraction using **MediaPipe**
- Supports **static images** (no live video or webcam needed)
- Classifies postures using **Random Forest Classifier**
- Trained on custom labeled images (sitting, standing, etc.)
- Supports **offline image prediction**

---
🧠 How It Works
Uses MediaPipe to detect 33 human pose landmarks
Extracts x and y coordinates for each keypoint
Trains a classifier using labeled landmark data
Predicts the posture based on test image landmarks

📈 Sample Result
Predicted posture: sitting


📚 Libraries Used
MediaPipe - Pose estimation
OpenCV - Image loading and preprocessing
scikit-learn - ML model and training
NumPy - Array operations
