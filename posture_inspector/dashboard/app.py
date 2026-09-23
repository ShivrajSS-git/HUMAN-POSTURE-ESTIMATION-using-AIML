import sys
import os

sys.modules['tensorflow'] = None

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

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
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "VIDEO FEED OFFLINE", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            time.sleep(0.03)

        processed_frame, _ = tr.process_frame(frame, draw_hud=True, plc_dispatcher=plc)

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
        <title>Industrial AI Ergonomics & Posture Inspector</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        
        <style>
            :root {
                --bg-main: #060913;
                --card-bg: rgba(15, 23, 42, 0.75);
                --card-border: rgba(255, 255, 255, 0.08);
                --accent-cyan: #00f2fe;
                --accent-emerald: #10b981;
                --accent-amber: #f59e0b;
                --accent-rose: #f43f5e;
                --text-primary: #f8fafc;
                --text-muted: #94a3b8;
                --mono-font: 'JetBrains Mono', monospace;
                --main-font: 'Plus Jakarta Sans', sans-serif;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; }
            
            body {
                background-color: var(--bg-main);
                background-image: 
                    radial-gradient(at 0% 0%, rgba(0, 242, 254, 0.08) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.06) 0px, transparent 50%);
                color: var(--text-primary);
                font-family: var(--main-font);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            header {
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(12px);
                border-bottom: 1px solid var(--card-border);
                padding: 16px 36px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .logo-group {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .logo-icon {
                width: 42px;
                height: 42px;
                background: linear-gradient(135deg, #00f2fe, #4facfe);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.4rem;
                box-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
            }

            h1 {
                font-size: 1.35rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                background: linear-gradient(90deg, #ffffff, #cbd5e1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .status-pill {
                display: flex;
                align-items: center;
                gap: 8px;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: var(--accent-emerald);
                padding: 6px 14px;
                border-radius: 30px;
                font-size: 0.82rem;
                font-weight: 600;
            }

            .pulse-dot {
                width: 8px;
                height: 8px;
                background-color: var(--accent-emerald);
                border-radius: 50%;
                box-shadow: 0 0 10px var(--accent-emerald);
                animation: pulse 1.8s infinite;
            }

            @keyframes pulse {
                0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
                100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }

            .main-layout {
                display: grid;
                grid-template-columns: 2.2fr 1fr;
                gap: 24px;
                padding: 28px 36px;
                flex-grow: 1;
            }

            .card {
                background: var(--card-bg);
                backdrop-filter: blur(16px);
                border: 1px solid var(--card-border);
                border-radius: 20px;
                padding: 24px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            }

            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 18px;
            }

            .card-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #e2e8f0;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            /* Framing Alert Banner */
            .framing-alert {
                display: none;
                background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05));
                border: 1px solid rgba(245, 158, 11, 0.4);
                color: #fbbf24;
                padding: 12px 18px;
                border-radius: 12px;
                margin-bottom: 16px;
                font-size: 0.88rem;
                font-weight: 600;
                align-items: center;
                gap: 10px;
                box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
            }

            .framing-alert.visible {
                display: flex;
                animation: fadeIn 0.4s ease;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-6px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .video-viewport {
                position: relative;
                width: 100%;
                border-radius: 16px;
                overflow: hidden;
                background: #000;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.8);
            }

            .video-viewport img {
                width: 100%;
                height: auto;
                max-height: 520px;
                display: block;
                object-fit: contain;
            }

            /* Metric Widgets */
            .metric-card {
                background: rgba(10, 15, 30, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 14px;
                padding: 16px 20px;
                margin-bottom: 14px;
                position: relative;
                overflow: hidden;
            }

            .metric-card::before {
                content: '';
                position: absolute;
                top: 0; left: 0; bottom: 0;
                width: 4px;
                background: var(--accent-cyan);
            }

            .metric-label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 1.2px;
                color: var(--text-muted);
                font-weight: 600;
            }

            .metric-value {
                font-family: var(--mono-font);
                font-size: 1.65rem;
                font-weight: 700;
                margin-top: 6px;
                color: #ffffff;
            }

            /* Progress Bar */
            .meter-container {
                margin-top: 8px;
                height: 6px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 10px;
                overflow: hidden;
            }

            .meter-fill {
                height: 100%;
                width: 0%;
                background: linear-gradient(90deg, #00f2fe, #4facfe);
                transition: width 0.4s ease;
                border-radius: 10px;
            }

            /* PLC Grid */
            .plc-matrix {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-top: 10px;
            }

            .plc-cell {
                background: rgba(10, 15, 30, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                padding: 12px;
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 0.8rem;
                font-weight: 600;
            }

            .led-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #334155;
                box-shadow: 0 0 4px rgba(0, 0, 0, 0.5);
                transition: all 0.3s ease;
            }

            .led-indicator.on {
                background: var(--accent-emerald);
                box-shadow: 0 0 12px var(--accent-emerald);
            }

            .led-indicator.warn {
                background: var(--accent-amber);
                box-shadow: 0 0 12px var(--accent-amber);
            }

            footer {
                padding: 16px 36px;
                text-align: center;
                border-top: 1px solid var(--card-border);
                background: rgba(15, 23, 42, 0.5);
                color: var(--text-muted);
                font-size: 0.82rem;
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo-group">
                <div class="logo-icon">🧍‍♂️</div>
                <div>
                    <h1>Posture Inspector AI</h1>
                    <span style="font-size: 0.75rem; color: var(--text-muted);">Industrial Ergonomics & Edge CV System</span>
                </div>
            </div>
            <div class="status-pill">
                <div class="pulse-dot"></div>
                <span>EDGE ENGINE OPERATIONAL</span>
            </div>
        </header>

        <div class="main-layout">
            <!-- Left Column: Video Viewport -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📹 Live Camera Feed & Augmented HUD</div>
                    <span style="font-size: 0.78rem; font-family: var(--mono-font); color: var(--accent-cyan);">MediaPipe 33 Landmark Mesh</span>
                </div>

                <div class="framing-alert" id="framing-alert">
                    <span>⚠️</span>
                    <span><strong>CAMERA FRAMING NOTICE:</strong> Step back from camera so your upper torso and shoulders are visible for accurate posture classification.</span>
                </div>

                <div class="video-viewport">
                    <img src="/video_feed" alt="Live Camera Posture Inspection Feed">
                </div>
            </div>

            <!-- Right Column: Real-time Telemetry -->
            <div style="display: flex; flex-direction: column; gap: 20px;">
                <div class="card">
                    <div class="card-title" style="margin-bottom: 16px;">📊 Inference Telemetry</div>

                    <div class="metric-card" style="border-left-color: var(--accent-cyan);">
                        <div class="metric-label">Current Posture</div>
                        <div class="metric-value" id="val-posture" style="color: var(--accent-cyan);">INITIALIZING...</div>
                        <div class="meter-container">
                            <div class="meter-fill" id="conf-fill"></div>
                        </div>
                    </div>

                    <div class="metric-card" id="card-risk">
                        <div class="metric-label">Ergonomic Risk Assessment</div>
                        <div class="metric-value" id="val-risk">LOW RISK</div>
                    </div>

                    <div class="metric-card" style="border-left-color: #a855f7;">
                        <div class="metric-label">Inference Latency</div>
                        <div class="metric-value" id="val-latency">0.0 ms</div>
                    </div>
                </div>

                <!-- Industrial PLC Pins -->
                <div class="card">
                    <div class="card-title" style="margin-bottom: 14px;">⚡ Industrial PLC Safety Relays</div>
                    <div class="plc-matrix" id="plc-matrix">
                        <div class="plc-cell">
                            <div class="led-indicator on"></div>
                            <span>NORMAL_OP</span>
                        </div>
                        <div class="plc-cell">
                            <div class="led-indicator"></div>
                            <span>ALARM_LIGHT</span>
                        </div>
                        <div class="plc-cell">
                            <div class="led-indicator"></div>
                            <span>BUZZER_ALERT</span>
                        </div>
                        <div class="plc-cell">
                            <div class="led-indicator"></div>
                            <span>CONVEYOR_HALT</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            Engineered with OpenCV, MediaPipe Pose, PyTorch & FastAPI | High-Performance Computer Vision Edge Solution
        </footer>

        <script>
            async function pollTelemetry() {
                try {
                    const res = await fetch('/api/telemetry');
                    const data = await res.json();

                    if (data && data.operator_posture) {
                        const posture = data.operator_posture.toUpperCase();
                        const conf = (data.model_confidence * 100).toFixed(0);
                        const risk = data.ergonomic_risk;
                        const latency = data.inference_latency_ms.toFixed(1);

                        // Update Posture text & meter
                        document.getElementById('val-posture').innerText = posture + ` (${conf}%)`;
                        document.getElementById('conf-fill').style.width = conf + '%';

                        // Check framing alert (too close to camera)
                        const alertBox = document.getElementById('framing-alert');
                        if (posture.includes('STEP BACK') || risk === 'FRAMING WARNING') {
                            alertBox.classList.add('visible');
                        } else {
                            alertBox.classList.remove('visible');
                        }

                        // Update Risk Card
                        const riskEl = document.getElementById('val-risk');
                        riskEl.innerText = risk;
                        if (risk === 'HIGH RISK') {
                            riskEl.style.color = 'var(--accent-rose)';
                        } else if (risk === 'MODERATE RISK' || risk === 'FRAMING WARNING') {
                            riskEl.style.color = 'var(--accent-amber)';
                        } else {
                            riskEl.style.color = 'var(--accent-emerald)';
                        }

                        // Latency readout
                        document.getElementById('val-latency').innerText = `${latency} ms`;

                        // PLC Pins rendering
                        const pins = data.plc_digital_outputs || {};
                        let plcHtml = '';
                        for (const [pin, val] of Object.entries(pins)) {
                            const pinName = pin.replace('PLC_OUT_', '');
                            const ledClass = val === 1 ? (pinName.includes('ALARM') || pinName.includes('HALT') ? 'led-indicator warn' : 'led-indicator on') : 'led-indicator';
                            plcHtml += `
                                <div class="plc-cell">
                                    <div class="${ledClass}"></div>
                                    <span>${pinName}</span>
                                </div>
                            `;
                        }
                        document.getElementById('plc-matrix').innerHTML = plcHtml;
                    }
                } catch(e) {}
            }
            setInterval(pollTelemetry, 400);
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
    print(f"[*] Launching Live Web Dashboard at: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_dashboard()
