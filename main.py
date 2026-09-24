import sys
import os

sys.modules['tensorflow'] = None

# Ensure project root is in sys.path for serverless execution
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    import argparse
    import cv2
    from posture_inspector.tracker import PoseTracker
    from posture_inspector.edge_integration.plc_dispatcher import IndustrialPLCDispatcher
    from posture_inspector.dashboard.app import app as dashboard_app

    # Top-level FastAPI instance for Vercel deployment & CLI entry point
    app = dashboard_app
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    app = FastAPI(title="Serverless Startup Error")
    err_msg = traceback.format_exc()

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    def catch_all(full_path: str = ""):
        return HTMLResponse(
            f"<html><body style='background:#0b0f19;color:#f87171;font-family:sans-serif;padding:30px;'>"
            f"<h2>⚠️ Serverless Startup Error</h2>"
            f"<pre style='background:#020617;border:1px solid #334155;color:#fca5a5;padding:20px;border-radius:8px;overflow-x:auto;'>{err_msg}</pre>"
            f"</body></html>",
            status_code=500
        )

def run_webcam_mode(camera_id=0):
    """Run real-time posture inspection on local webcam feed."""
    print("==================================================")
    print(" [*] Starting Real-Time OpenCV & MediaPipe Pose Inspector")
    print(" [*] Press 'q' in the window to Exit, 's' to Save Snapshot")
    print("==================================================")

    tracker = PoseTracker(static_image_mode=False)
    plc = IndustrialPLCDispatcher()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[!] Unable to open camera {camera_id}. Checking sample test image...")
        test_img_path = "HUMAN POSTURE ESTIMATION/test_images/test_standing.jpeg"
        if os.path.exists(test_img_path):
            run_test_image_mode(test_img_path)
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to read video frame.")
            break

        processed_frame, info = tracker.process_frame(frame, draw_hud=True, plc_dispatcher=plc)

        cv2.imshow("Industrial Human Posture AI Inspector (Press 'q' to exit)", processed_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snapshot_path = "snapshot_result.png"
            cv2.imwrite(snapshot_path, processed_frame)
            print(f"[SAVE] Saved snapshot to {snapshot_path}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


def run_test_image_mode(image_path):
    """Run posture prediction on static image file."""
    print("==================================================")
    print(f" [*] Processing Test Image: {image_path}")
    print("==================================================")

    if not os.path.exists(image_path):
        print(f"[!] Image path not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Failed to load image from {image_path}")
        return

    tracker = PoseTracker(static_image_mode=True)
    plc = IndustrialPLCDispatcher()

    processed_frame, info = tracker.process_frame(img, draw_hud=True, plc_dispatcher=plc)

    print("\n--- Inspection Summary ---")
    print(f" Detected: {info['pose_detected']}")
    print(f" Posture Prediction: {info['posture_label'].upper()}")
    print(f" Model Confidence: {info['confidence']*100:.1f}%")
    print(f" Ergonomic Strain Risk: {info['ergonomic_risk']}")
    print(f" Latency: {info['latency_ms']:.1f} ms")
    print(f" PLC Output: {info['plc_state'].get('plc_digital_outputs', {})}")
    print("--------------------------\n")

    output_path = "output_inspection_result.png"
    cv2.imwrite(output_path, processed_frame)
    print(f"[SAVE] Saved annotated result image to: {output_path}")

    tracker.close()


def main():
    parser = argparse.ArgumentParser(description="Industrial Human Posture AI Inspection System")
    parser.add_argument("--mode", type=str, default="test-image",
                        choices=["webcam", "dashboard", "test-image", "train"],
                        help="Execution mode (default: test-image)")
    parser.add_argument("--image", type=str, default="HUMAN POSTURE ESTIMATION/test_images/test_standing.jpeg",
                        help="Path to test image file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--port", type=int, default=8000, help="Web Dashboard Port (default: 8000)")

    args = parser.parse_args()

    if args.mode == "webcam":
        run_webcam_mode(camera_id=args.camera)
    elif args.mode == "dashboard":
        from posture_inspector.dashboard.app import run_dashboard
        run_dashboard(port=args.port)
    elif args.mode == "train":
        from posture_inspector.models.train import train_models
        train_models()
    elif args.mode == "test-image":
        run_test_image_mode(args.image)

if __name__ == "__main__":
    main()
