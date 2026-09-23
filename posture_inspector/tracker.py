import sys
sys.modules['tensorflow'] = None

import time
import cv2
import mediapipe as mp
import numpy as np

from posture_inspector.feature_extractor import FeatureExtractor
from posture_inspector.models.classifier import PostureClassifierManager

class PoseTracker:
    """
    OpenCV & MediaPipe Real-Time Human Pose Tracking Engine.
    Performs 33 keypoint landmark detection, joint angle calculations, HUD visualization,
    and ML/DL model inference.
    """
    def __init__(self, static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.classifier = PostureClassifierManager()
        self.prev_time = time.time()
        self.fps = 0.0
        self.latency_ms = 0.0

    def process_frame(self, frame, draw_hud=True, plc_dispatcher=None):
        """
        Process a single OpenCV BGR image frame.
        
        Returns:
            processed_frame (np.ndarray), result_dict (dict)
        """
        start_t = time.time()
        
        # Calculate FPS
        curr_t = time.time()
        dt = curr_t - self.prev_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_time = curr_t

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        result_info = {
            "pose_detected": False,
            "landmarks": None,
            "posture_label": "No Person Detected",
            "confidence": 0.0,
            "ergonomic_risk": "Low",
            "metrics": {},
            "fps": round(self.fps, 1),
            "latency_ms": 0.0,
            "plc_state": {}
        }

        if results.pose_landmarks:
            result_info["pose_detected"] = True
            landmarks = results.pose_landmarks.landmark
            result_info["landmarks"] = landmarks

            # Draw MediaPipe landmark skeleton
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Extract metrics & feature vector
            metrics = FeatureExtractor.extract_metrics(landmarks)
            result_info["metrics"] = metrics
            feature_vector = FeatureExtractor.extract_feature_vector(landmarks)

            # Check Camera Distance / Framing (e.g., face only visible)
            framing_status = FeatureExtractor.check_camera_framing(landmarks)
            result_info["framing_status"] = framing_status

            if framing_status == "TOO_CLOSE_STEP_BACK":
                posture_label = "STEP BACK (TOO CLOSE)"
                confidence = 0.99
                ergonomic_risk = "FRAMING WARNING"
                result_info["posture_label"] = posture_label
                result_info["confidence"] = confidence
                result_info["ergonomic_risk"] = ergonomic_risk
            else:
                # Model Inference
                posture_label, confidence = self.classifier.predict(feature_vector)
                result_info["posture_label"] = posture_label
                result_info["confidence"] = confidence

                # Calculate Ergonomic Risk Level based on posture & neck inclination
                neck_inc = metrics.get("neck_inclination", 0)
                torso_angle = metrics.get("torso_spine_angle", 0)

                if posture_label in ["slouching", "bending"] or neck_inc > 30.0 or torso_angle > 30.0:
                    ergonomic_risk = "HIGH RISK"
                elif neck_inc > 18.0 or torso_angle > 18.0:
                    ergonomic_risk = "MODERATE RISK"
                else:
                    ergonomic_risk = "LOW RISK"

                result_info["ergonomic_risk"] = ergonomic_risk

            # Dispatch PLC Industrial Signal if dispatcher provided
            if plc_dispatcher is not None:
                plc_state = plc_dispatcher.dispatch_signal(
                    posture_label=posture_label,
                    confidence=confidence,
                    ergonomic_risk=ergonomic_risk,
                    latency_ms=self.latency_ms
                )
                result_info["plc_state"] = plc_state

            # Draw HUD Overlays if requested
            if draw_hud:
                self._draw_hud(frame, result_info, metrics)

        else:
            if draw_hud:
                cv2.putText(frame, "STATUS: SEARCHING FOR PERSON...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        self.latency_ms = (time.time() - start_t) * 1000.0
        result_info["latency_ms"] = round(self.latency_ms, 1)

        return frame, result_info

    def _draw_hud(self, frame, result_info, metrics):
        """
        Draw rich augmented reality HUD graphics on OpenCV frame.
        """
        h, w, _ = frame.shape
        label = result_info["posture_label"].upper()
        conf = result_info["confidence"]
        risk = result_info["ergonomic_risk"]
        fps = result_info["fps"]

        # Color palette
        if risk in ["HIGH RISK", "FRAMING WARNING"]:
            color = (0, 0, 255) if risk == "HIGH RISK" else (0, 165, 255) # Red or Orange
        elif risk == "MODERATE RISK":
            color = (0, 165, 255) # Orange
        else:
            color = (0, 255, 0) # Green

        # Top Banner Panel
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        
        # Posture & Risk Header
        cv2.putText(frame, f"POSTURE: {label} ({conf*100:.0f}%)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"RISK: {risk}", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Performance Metrics Header
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 180, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"LATENCY: {self.latency_ms:.1f}ms", (w - 180, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Framing Warning Overlay if camera is too close
        if result_info.get("framing_status") == "TOO_CLOSE_STEP_BACK":
            cv2.rectangle(frame, (20, 90), (w - 20, 140), (0, 140, 255), -1)
            cv2.putText(frame, "WARNING: CAMERA TOO CLOSE! STEP BACK FOR FULL TORSO VIEW",
                        (30, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Bottom Joint Angle Overlay Panel
        if metrics:
            panel_y = h - 90
            cv2.rectangle(frame, (0, panel_y), (w, h), (10, 10, 10), -1)
            
            torso = metrics.get('torso_spine_angle', 0)
            neck = metrics.get('neck_inclination', 0)
            knee = metrics.get('avg_knee_angle', 0)
            hip = metrics.get('avg_hip_angle', 0)

            angle_txt1 = f"Torso: {torso:.1f}deg  |  Neck: {neck:.1f}deg"
            angle_txt2 = f"Avg Knee: {knee:.1f}deg  |  Avg Hip: {hip:.1f}deg"

            cv2.putText(frame, angle_txt1, (20, panel_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(frame, angle_txt2, (20, panel_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    def close(self):
        self.pose.close()
