import numpy as np

class FeatureExtractor:
    """
    Biomechanical Feature Extraction Engine for MediaPipe 33 Landmark Pose Data.
    Extracts scale-normalized coordinates and joint angles for ergonomics & posture analysis.
    """
    
    # MediaPipe landmark indices
    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculate 2D angle (in degrees) at point 'b' given points a, b, c.
        Points can be 2D tuples/lists or numpy arrays.
        """
        a = np.array(a[:2])
        b = np.array(b[:2])
        c = np.array(c[:2])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

    @staticmethod
    def calculate_spine_inclination(shoulder_mid, hip_mid):
        """
        Calculate spine inclination angle relative to vertical axis (0 deg = straight vertical).
        """
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        if dy == 0:
            return 90.0
        angle = np.abs(np.arctan2(dx, -dy) * 180.0 / np.pi)
        return angle

    @classmethod
    def extract_metrics(cls, landmarks):
        """
        Extract ergonomic joint angles and posture metrics from MediaPipe landmarks list.
        
        Parameters:
            landmarks: list of MediaPipe landmark objects (containing .x, .y, .z, .visibility)
        
        Returns:
            dict of metric angles & ratios
        """
        pts = {i: np.array([lm.x, lm.y, lm.z]) for i, lm in enumerate(landmarks)}

        left_shoulder = pts[cls.LEFT_SHOULDER]
        right_shoulder = pts[cls.RIGHT_SHOULDER]
        left_hip = pts[cls.LEFT_HIP]
        right_hip = pts[cls.RIGHT_HIP]
        left_knee = pts[cls.LEFT_KNEE]
        right_knee = pts[cls.RIGHT_KNEE]
        left_ankle = pts[cls.LEFT_ANKLE]
        right_ankle = pts[cls.RIGHT_ANKLE]

        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        hip_mid = (left_hip + right_hip) / 2.0
        ear_mid = (pts[cls.LEFT_EAR] + pts[cls.RIGHT_EAR]) / 2.0

        # Angles
        torso_spine_angle = cls.calculate_spine_inclination(shoulder_mid, hip_mid)
        left_knee_angle = cls.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = cls.calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = cls.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = cls.calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Neck forward inclination (Ear to Shoulder angle vs vertical)
        neck_inclination = cls.calculate_spine_inclination(ear_mid, shoulder_mid)

        # Shoulder tilt horizontal delta
        shoulder_tilt = np.abs(left_shoulder[1] - right_shoulder[1]) * 100.0

        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2.0

        return {
            "torso_spine_angle": float(torso_spine_angle),
            "left_knee_angle": float(left_knee_angle),
            "right_knee_angle": float(right_knee_angle),
            "avg_knee_angle": float(avg_knee_angle),
            "left_hip_angle": float(left_hip_angle),
            "right_hip_angle": float(right_hip_angle),
            "avg_hip_angle": float(avg_hip_angle),
            "neck_inclination": float(neck_inclination),
            "shoulder_tilt": float(shoulder_tilt),
        }

    @classmethod
    def extract_feature_vector(cls, landmarks):
        """
        Extract scale-invariant feature vector for ML/DL classifier input.
        
        Output vector size:
        33 keypoints * (x, y, z) normalized relative to hip center + 9 metric features = 108 features.
        """
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Center normalization relative to hip center
        hip_center = (pts[cls.LEFT_HIP] + pts[cls.RIGHT_HIP]) / 2.0
        norm_pts = pts - hip_center

        # Scale normalization relative to torso height
        torso_height = np.linalg.norm(
            ((pts[cls.LEFT_SHOULDER] + pts[cls.RIGHT_SHOULDER]) / 2.0) - hip_center
        )
        if torso_height > 1e-4:
            norm_pts /= torso_height

        # Flatten normalized landmarks (33 * 3 = 99 features)
        flat_coords = norm_pts.flatten()

        # Biomechanical metrics (9 features)
        metrics = cls.extract_metrics(landmarks)
        metric_values = np.array([
            metrics["torso_spine_angle"] / 90.0,
            metrics["left_knee_angle"] / 180.0,
            metrics["right_knee_angle"] / 180.0,
            metrics["avg_knee_angle"] / 180.0,
            metrics["left_hip_angle"] / 180.0,
            metrics["right_hip_angle"] / 180.0,
            metrics["avg_hip_angle"] / 180.0,
            metrics["neck_inclination"] / 90.0,
            metrics["shoulder_tilt"] / 10.0,
        ])

        return np.concatenate([flat_coords, metric_values])
