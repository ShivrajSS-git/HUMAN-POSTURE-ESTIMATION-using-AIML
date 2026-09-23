"""
Human Posture Estimation & Ergonomics Inspector - Entry Script.
Upgraded to use posture_inspector PyTorch/OpenCV pipeline.
"""
import sys
sys.modules['tensorflow'] = None

import os
import cv2
from posture_inspector.tracker import PoseTracker
from posture_inspector.edge_integration.plc_dispatcher import IndustrialPLCDispatcher

def main():
    image_path = "HUMAN POSTURE ESTIMATION/test_images/test_standing.jpeg"
    if not os.path.exists(image_path):
        print(f"[!] Test image not found at: {image_path}")
        return

    img = cv2.imread(image_path)
    tracker = PoseTracker(static_image_mode=True)
    plc = IndustrialPLCDispatcher()

    processed_frame, info = tracker.process_frame(img, draw_hud=True, plc_dispatcher=plc)

    print("==================================================")
    print(" 🧍‍♂️ Human Posture AI Inspection Output")
    print("==================================================")
    print(f" Posture Prediction: {info['posture_label'].upper()}")
    print(f" Model Confidence: {info['confidence']*100:.1f}%")
    print(f" Ergonomic Strain Risk: {info['ergonomic_risk']}")
    print(f" Latency: {info['latency_ms']:.1f} ms")
    print(f" PLC Signals: {info['plc_state'].get('plc_digital_outputs', {})}")
    print("==================================================")

    output_path = "HUMAN POSTURE ESTIMATION/inspection_result.png"
    cv2.imwrite(output_path, processed_frame)
    print(f"[*] Saved annotated output to: {output_path}")

    tracker.close()

if __name__ == "__main__":
    main()
