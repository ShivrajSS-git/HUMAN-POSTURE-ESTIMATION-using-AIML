# Importing required libraries:
# - cv2: for image reading and processing
# - mediapipe: for extracting human pose landmarks
# - numpy: for numerical operations and array handling
# - joblib: for saving/loading trained ML models
# - RandomForestClassifier: ML model used to classify human posture

import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
# Initialize MediaPipe drawing and pose detection modules:
# - mp_drawing: for visualizing pose landmarks on images
# - mp_pose: provides the pose estimation model and utilities

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to extract pose landmarks from an input image
def extract_pose_landmarks(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [l.x for l in landmarks] + [l.y for l in landmarks]
            return keypoints
        return None

  import os

data_dir = "HUMAN POSTURE ESTIMATION/images_dataset"  # folder with subfolders like 'sitting', 'standing'
X, y = [], []

for label in os.listdir(data_dir):
    folder = os.path.join(data_dir, label)
    for img_name in os.listdir(folder):
        path = os.path.join(folder,"sitting1.jpeg")
        img = cv2.imread("HUMAN POSTURE ESTIMATION/images_dataset/sitting/sitting1.jpeg")
        keypoints = extract_pose_landmarks(img)
        if keypoints:
            X.append(keypoints)
            y.append(label)

X = np.array(X)
y = np.array(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))
joblib.dump(clf, "posture_model.pkl")

image_path = "HUMAN POSTURE ESTIMATION/test_images/test_standing.jpeg"
img = cv2.imread(image_path)
keypoints = extract_pose_landmarks(img)

if keypoints:
    clf = joblib.load("posture_model.pkl")
    pred = clf.predict([keypoints])
    print("Predicted posture:", pred[0])
    
    cv2.putText(img, f"Posture: {pred[0]}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
