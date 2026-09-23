import sys
sys.modules['tensorflow'] = None

import os
import time
import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

from posture_inspector.tracker import PoseTracker
from posture_inspector.edge_integration.plc_dispatcher import IndustrialPLCDispatcher

app = FastAPI(title="Industrial Human Posture Inspection System", version="2.0.0")

# Global singleton engine instances
tracker = None
plc_dispatcher = None

def get_engine():
    global tracker, plc_dispatcher
    if tracker is None:
        tracker = PoseTracker(static_image_mode=False)
    if plc_dispatcher is None:
        plc_dispatcher = IndustrialPLCDispatcher()
    return tracker, plc_dispatcher


def generate_frames():
    """MJPEG Video Stream Generator."""
    tr, plc = get_engine()
    
    # Try opening physical camera, fallback to synthetic test frame if unavailable
    cap = cv2.VideoCapture(0)
    camera_available = cap.isOpened()
    
    test_img_path = "HUMAN POSTURE ESTIMATION/test_images/test_standing.jpeg"
    test_img = None
    if not camera_available and os.path.exists(test_img_path):
        test_img = cv2.imread(test_img_path)

    while True:
        if camera_available:
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            if test_img is not None:
                frame = test_img.copy()
            else:
                # Blank placeholder canvas if no test image
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "VIDEO FEED OFFLINE", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            time.sleep(0.03) # ~30 FPS frame timing simulation

        # Process frame through PoseTracker & PLC Dispatcher
        processed_frame, _ = tr.process_frame(frame, draw_hud=True, plc_dispatcher=plc)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if camera_available:
        cap.release()


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Industrial Posture AI Inspection Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
            body { background-color: #0b0f19; color: #f1f5f9; display: flex; flex-direction: column; min-height: 100vh; }
            header { background: linear-gradient(90deg, #1e293b, #0f172a); border-bottom: 1px solid #334155; padding: 18px 30px; display: flex; justify-content: space-between; align-items: center; }
            h1 { font-size: 1.5rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 10px; }
            .badge { background: #0284c7; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; }
            .container { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; padding: 25px; flex-grow: 1; }
            .card { background: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            .video-container { position: relative; width: 100%; border-radius: 10px; overflow: hidden; background: #000; display: flex; justify-content: center; }
            .video-stream { width: 100%; height: auto; max-height: 520px; object-fit: contain; }
            .stat-box { background: #0f172a; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #38bdf8; }
            .stat-title { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
            .stat-value { font-size: 1.6rem; font-weight: 700; margin-top: 5px; color: #f8fafc; }
            .plc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
            .plc-pin { background: #0f172a; padding: 10px; border-radius: 8px; text-align: center; font-size: 0.8rem; border: 1px solid #334155; }
            .pin-on { border-color: #22c55e; color: #4ade80; background: rgba(34, 197, 94, 0.1); }
            .pin-off { border-color: #64748b; color: #94a3b8; }
            footer { text-align: center; padding: 15px; background: #0f172a; border-top: 1px solid #334155; color: #64748b; font-size: 0.85rem; }
        </style>
    </head>
    <body>
        <header>
            <h1>🧍‍♂️ Human Posture AI Inspector <span class="badge">LIVE CV/EDGE PIPELINE</span></h1>
            <div style="color: #4ade80; font-size: 0.9rem; font-weight: 600;">● System Operational</div>
        </header>

        <div class="container">
            <div class="card">
                <h3 style="margin-bottom: 15px; color: #cbd5e1;">Live Video Stream & Posture HUD Overlay</h3>
                <div class="video-container">
                    <img class="video-stream" src="/video_feed" alt="Live Camera Posture Inspection Feed">
                </div>
            </div>

            <div style="display: flex; flex-direction: column; gap: 20px;">
                <div class="card">
                    <h3 style="margin-bottom: 15px; color: #cbd5e1;">Real-Time Telemetry</h3>
                    <div class="stat-box">
                        <div class="stat-title">Current Posture</div>
                        <div class="stat-value" id="val-posture" style="color: #38bdf8;">Loading...</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-title">Ergonomic Strain Risk</div>
                        <div class="stat-value" id="val-risk" style="color: #4ade80;">LOW RISK</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-title">Inference Speed & FPS</div>
                        <div class="stat-value" id="val-performance">0.0 FPS / 0.0 ms</div>
                    </div>
                </div>

                <div class="card">
                    <h3 style="margin-bottom: 10px; color: #cbd5e1;">Industrial PLC Relay Signals</h3>
                    <div class="plc-grid" id="plc-pins">
                        <div class="plc-pin pin-on">NORMAL_OP: 1</div>
                        <div class="plc-pin pin-off">ALARM_LIGHT: 0</div>
                        <div class="plc-pin pin-off">BUZZER_ALERT: 0</div>
                        <div class="plc-pin pin-off">CONVEYOR_HALT: 0</div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            Built with OpenCV, MediaPipe, PyTorch & FastAPI | Edge AI Computer Vision Solution
        </footer>

        <script>
            async function updateTelemetry() {
                try {
                    const res = await fetch('/api/telemetry');
                    const data = await res.json();
                    if (data && data.operator_posture) {
                        document.getElementById('val-posture').innerText = data.operator_posture.toUpperCase() + ` (${(data.model_confidence*100).toFixed(0)}%)`;
                        document.getElementById('val-risk').innerText = data.ergonomic_risk;
                        document.getElementById('val-risk').style.color = data.ergonomic_risk === 'HIGH RISK' ? '#ef4444' : (data.ergonomic_risk === 'MODERATE RISK' ? '#f97316' : '#4ade80');
                        document.getElementById('val-performance').innerText = `${data.inference_latency_ms.toFixed(1)} ms latency`;
                        
                        const pins = data.plc_digital_outputs || {};
                        let html = '';
                        for (const [k, v] of Object.entries(pins)) {
                            const cls = v === 1 ? 'pin-on' : 'pin-off';
                            html += `<div class="plc-pin ${cls}">${k.replace('PLC_OUT_', '')}: ${v}</div>`;
                        }
                        document.getElementById('plc-pins').innerHTML = html;
                    }
                } catch(e) {}
            }
            setInterval(updateTelemetry, 500);
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/video_feed")
def video_feed():
    """Video streaming route."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/telemetry")
def get_telemetry():
    """REST API endpoint returning latest JSON telemetry."""
    _, plc = get_engine()
    if plc.last_state:
        return plc.last_state
    return {"operator_posture": "standing", "model_confidence": 0.95, "ergonomic_risk": "LOW RISK", "inference_latency_ms": 12.4, "plc_digital_outputs": {"PLC_OUT_NORMAL_OP": 1, "PLC_OUT_ALARM_LIGHT": 0, "PLC_OUT_BUZZER_ALERT": 0, "PLC_OUT_CONVEYOR_HALT": 0}}


def run_dashboard(host="127.0.0.1", port=8000):
    """Launch Uvicorn server for Web Dashboard."""
    print(f"🚀 Launching Live Web Dashboard at: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_dashboard()
