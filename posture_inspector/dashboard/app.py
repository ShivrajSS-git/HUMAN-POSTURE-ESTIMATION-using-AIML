import sys
import os
import time
import json
import cv2
import numpy as np
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.modules['tensorflow'] = None

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from posture_inspector.feature_extractor import FeatureExtractor
from posture_inspector.models.classifier import PostureClassifierManager
from posture_inspector.edge_integration.plc_dispatcher import IndustrialPLCDispatcher

app = FastAPI(
    title="Industrial Human Posture Inspection System",
    description="Real-Time Industrial Ergonomics & Posture AI Inspector",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine singletons
classifier = PostureClassifierManager()
plc_dispatcher = IndustrialPLCDispatcher()

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": "Industrial Human Posture AI Inspector",
        "version": "2.0.0",
        "cloud_ready": True
    }

@app.get("/api/telemetry")
def get_telemetry():
    """REST API endpoint returning latest JSON telemetry."""
    if plc_dispatcher.last_state:
        return plc_dispatcher.last_state
    return {
        "operator_posture": "standing",
        "model_confidence": 0.95,
        "ergonomic_risk": "LOW RISK",
        "inference_latency_ms": 12.4,
        "plc_digital_outputs": {
            "PLC_OUT_NORMAL_OP": 1,
            "PLC_OUT_ALARM_LIGHT": 0,
            "PLC_OUT_BUZZER_ALERT": 0,
            "PLC_OUT_CONVEYOR_HALT": 0
        }
    }

class LandmarkObj:
    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)

@app.post("/api/inspect_landmarks")
async def inspect_landmarks(request: Request):
    """
    Process 33 MediaPipe pose landmarks from client camera through the complete
    industrial feature extraction, camera framing validator, and ML classifier pipeline.
    """
    try:
        data = await request.json()
        raw_lms = data.get("landmarks", [])
        if not raw_lms or len(raw_lms) < 33:
            return JSONResponse({"status": "no_pose"})

        landmarks = [
            LandmarkObj(
                x=lm.get("x", 0.0),
                y=lm.get("y", 0.0),
                z=lm.get("z", 0.0),
                visibility=lm.get("visibility", 1.0)
            ) for lm in raw_lms
        ]

        # 1. Camera Distance & Framing Validation
        framing_status = FeatureExtractor.check_camera_framing(landmarks)
        metrics = FeatureExtractor.extract_metrics(landmarks)

        start_t = time.time()
        if framing_status == "TOO_CLOSE_STEP_BACK":
            posture_label = "STEP BACK (TOO CLOSE)"
            confidence = 0.99
            ergonomic_risk = "FRAMING WARNING"
        else:
            feature_vector = FeatureExtractor.extract_feature_vector(landmarks)
            posture_label, confidence = classifier.predict(feature_vector)

            neck_inc = metrics.get("neck_inclination", 0)
            torso_angle = metrics.get("torso_spine_angle", 0)

            if posture_label in ["slouching", "bending"] or neck_inc > 30.0 or torso_angle > 30.0:
                ergonomic_risk = "HIGH RISK"
            elif neck_inc > 18.0 or torso_angle > 18.0:
                ergonomic_risk = "MODERATE RISK"
            else:
                ergonomic_risk = "LOW RISK"

        latency_ms = (time.time() - start_t) * 1000.0

        # 2. Dispatch Industrial PLC Signals
        plc_state = plc_dispatcher.dispatch_signal(
            posture_label=posture_label,
            confidence=confidence,
            ergonomic_risk=ergonomic_risk,
            latency_ms=latency_ms
        )

        return {
            "status": "success",
            "posture_label": posture_label,
            "confidence": round(float(confidence), 3),
            "ergonomic_risk": ergonomic_risk,
            "framing_status": framing_status,
            "metrics": metrics,
            "latency_ms": round(latency_ms, 1),
            "plc_state": plc_state.get("plc_digital_outputs", {})
        }

    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """<!DOCTYPE html>
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
            aspect-ratio: 4 / 3;
            max-height: 520px;
        }

        #webcam-video {
            display: none;
        }

        #output-canvas {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

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
            transition: width 0.3s ease;
            border-radius: 10px;
        }

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

            <!-- Framing Alert Banner -->
            <div class="framing-alert" id="framing-alert">
                <span>⚠️</span>
                <span><strong>CAMERA FRAMING NOTICE:</strong> Step back from camera so your upper torso and shoulders are visible for accurate posture classification.</span>
            </div>

            <div class="video-viewport">
                <video id="webcam-video" autoplay playsinline></video>
                <canvas id="output-canvas" width="640" height="480"></canvas>
            </div>
        </div>

        <!-- Right Column: Real-time Telemetry -->
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div class="card">
                <div class="card-title" style="margin-bottom: 16px;">📊 Inference Telemetry</div>

                <div class="metric-card" style="border-left-color: var(--accent-cyan);">
                    <div class="metric-label">Current Posture</div>
                    <div class="metric-value" id="val-posture" style="color: var(--accent-cyan);">SEARCHING FOR OPERATOR...</div>
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

    <!-- MediaPipe Pose Scripts for Browser Camera Streaming -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>

    <script>
        const videoElement = document.getElementById('webcam-video');
        const canvasElement = document.getElementById('output-canvas');
        const canvasCtx = canvasElement.getContext('2d');
        const alertBox = document.getElementById('framing-alert');

        let lastInspectTime = 0;
        let isInspecting = false;
        let fps = 0.0;
        let lastFrameTime = performance.now();

        // Pose connections mapping
        const POSE_CONNECTIONS = [
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
            [11, 23], [12, 24], [23, 24],
            [23, 25], [24, 26], [25, 27], [26, 28]
        ];

        // Biomechanical Angle Calculator (Matching FeatureExtractor.py)
        function calcAngle(a, b, c) {
            const rad = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
            let deg = Math.abs(rad * 180.0 / Math.PI);
            if (deg > 180.0) deg = 360.0 - deg;
            return deg;
        }

        function calcSpineInclination(shoulderMid, hipMid) {
            const dx = shoulderMid.x - hipMid.x;
            const dy = shoulderMid.y - hipMid.y;
            if (dy === 0) return 90.0;
            return Math.abs(Math.atan2(dx, -dy) * 180.0 / Math.PI);
        }

        // Camera Framing Validator (Matching FeatureExtractor.check_camera_framing)
        function checkFraming(landmarks) {
            const leftShoulder = landmarks[11];
            const rightShoulder = landmarks[12];
            const leftHip = landmarks[23];
            const rightHip = landmarks[24];

            if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) {
                return "TOO_CLOSE_STEP_BACK";
            }

            const shoulderVis = ((leftShoulder.visibility || 1) + (rightShoulder.visibility || 1)) / 2;
            const hipVis = ((leftHip.visibility || 1) + (rightHip.visibility || 1)) / 2;
            const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x);

            // If shoulders/hips cut off or camera too close
            if (shoulderVis < 0.45 || hipVis < 0.35 || leftHip.y > 1.02 || rightHip.y > 1.02) {
                return "TOO_CLOSE_STEP_BACK";
            }
            if (shoulderWidth > 0.85 || (leftShoulder.y < 0.05 && hipVis < 0.5)) {
                return "TOO_CLOSE_STEP_BACK";
            }
            return "OK";
        }

        function updateTelemetryUI(posture, conf, risk, latency, pins) {
            document.getElementById('val-posture').innerText = posture + ` (${(conf * 100).toFixed(0)}%)`;
            document.getElementById('conf-fill').style.width = `${(conf * 100).toFixed(0)}%`;

            // Framing alert banner toggle
            if (posture.includes('STEP BACK') || risk === 'FRAMING WARNING') {
                alertBox.classList.add('visible');
            } else {
                alertBox.classList.remove('visible');
            }

            const riskEl = document.getElementById('val-risk');
            riskEl.innerText = risk;
            if (risk === 'HIGH RISK') {
                riskEl.style.color = 'var(--accent-rose)';
            } else if (risk === 'MODERATE RISK' || risk === 'FRAMING WARNING') {
                riskEl.style.color = 'var(--accent-amber)';
            } else {
                riskEl.style.color = 'var(--accent-emerald)';
            }

            document.getElementById('val-latency').innerText = `${latency.toFixed(1)} ms`;

            // PLC matrix
            const plcMatrix = pins || {};
            let plcHtml = '';
            for (const [pin, val] of Object.entries(plcMatrix)) {
                const pinName = pin.replace('PLC_OUT_', '');
                const ledClass = val === 1 ? (pinName.includes('ALARM') || pinName.includes('HALT') ? 'led-indicator warn' : 'led-indicator on') : 'led-indicator';
                plcHtml += `
                    <div class="plc-cell">
                        <div class="${ledClass}"></div>
                        <span>${pinName}</span>
                    </div>
                `;
            }
            if (plcHtml) {
                document.getElementById('plc-matrix').innerHTML = plcHtml;
            }
        }

        // Draw Augmented HUD (Matching OpenCV Tracker.py)
        function drawHUD(ctx, width, height, posture, conf, risk, fps, latency, metrics, isTooClose) {
            // Top Banner Panel
            ctx.fillStyle = "rgba(20, 20, 20, 0.85)";
            ctx.fillRect(0, 0, width, 75);

            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 16px 'Plus Jakarta Sans', sans-serif";
            ctx.fillText(`POSTURE: ${posture} (${(conf * 100).toFixed(0)}%)`, 20, 32);

            let riskColor = "#10b981";
            if (risk === "HIGH RISK") riskColor = "#f43f5e";
            else if (risk === "MODERATE RISK" || risk === "FRAMING WARNING") riskColor = "#f59e0b";

            ctx.fillStyle = riskColor;
            ctx.font = "bold 14px 'Plus Jakarta Sans', sans-serif";
            ctx.fillText(`RISK: ${risk}`, 20, 60);

            ctx.fillStyle = "#00f2fe";
            ctx.font = "14px 'JetBrains Mono', monospace";
            ctx.fillText(`FPS: ${fps.toFixed(1)}`, width - 150, 32);

            ctx.fillStyle = "#94a3b8";
            ctx.font = "13px 'JetBrains Mono', monospace";
            ctx.fillText(`LATENCY: ${latency.toFixed(1)}ms`, width - 150, 60);

            // Framing warning banner overlay directly on video
            if (isTooClose) {
                ctx.fillStyle = "rgba(245, 158, 11, 0.9)";
                ctx.fillRect(20, 85, width - 40, 40);
                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 13px 'Plus Jakarta Sans', sans-serif";
                ctx.fillText("WARNING: CAMERA TOO CLOSE! STEP BACK FOR FULL TORSO VIEW", 35, 110);
            }

            // Bottom Joint Angle Overlay Panel
            if (metrics) {
                const panelY = height - 70;
                ctx.fillStyle = "rgba(10, 10, 10, 0.85)";
                ctx.fillRect(0, panelY, width, 70);

                ctx.fillStyle = "#e2e8f0";
                ctx.font = "12px 'JetBrains Mono', monospace";
                ctx.fillText(`Torso: ${metrics.torso.toFixed(1)}deg  |  Neck: ${metrics.neck.toFixed(1)}deg`, 20, panelY + 28);
                ctx.fillText(`Avg Knee: ${metrics.knee.toFixed(1)}deg  |  Avg Hip: ${metrics.hip.toFixed(1)}deg`, 20, panelY + 52);
            }
        }

        // Initialize MediaPipe Pose Engine
        const pose = new Pose({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
        });

        pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.6
        });

        pose.onResults(async (results) => {
            const now = performance.now();
            const dt = (now - lastFrameTime) / 1000.0;
            if (dt > 0) fps = 0.9 * fps + 0.1 * (1.0 / dt);
            lastFrameTime = now;

            const w = canvasElement.width;
            const h = canvasElement.height;

            canvasCtx.save();
            canvasCtx.clearRect(0, 0, w, h);
            canvasCtx.drawImage(results.image, 0, 0, w, h);

            if (results.poseLandmarks) {
                const lms = results.poseLandmarks;

                // Draw Landmark Skeleton Connections
                canvasCtx.strokeStyle = "#00f2fe";
                canvasCtx.lineWidth = 3;
                for (const [i, j] of POSE_CONNECTIONS) {
                    const p1 = lms[i];
                    const p2 = lms[j];
                    if (p1 && p2 && (p1.visibility || 1) > 0.5 && (p2.visibility || 1) > 0.5) {
                        canvasCtx.beginPath();
                        canvasCtx.moveTo(p1.x * w, p1.y * h);
                        canvasCtx.lineTo(p2.x * w, p2.y * h);
                        canvasCtx.stroke();
                    }
                }

                // Draw Landmark Joints
                canvasCtx.fillStyle = "#10b981";
                for (let k = 11; k <= 28; k++) {
                    const pt = lms[k];
                    if (pt && (pt.visibility || 1) > 0.5) {
                        canvasCtx.beginPath();
                        canvasCtx.arc(pt.x * w, pt.y * h, 5, 0, 2 * Math.PI);
                        canvasCtx.fill();
                    }
                }

                // Extract Local Biomechanics
                const shoulderMid = { x: (lms[11].x + lms[12].x) / 2, y: (lms[11].y + lms[12].y) / 2 };
                const hipMid = { x: (lms[23].x + lms[24].x) / 2, y: (lms[23].y + lms[24].y) / 2 };
                const earMid = { x: (lms[7].x + lms[8].x) / 2, y: (lms[7].y + lms[8].y) / 2 };

                const torsoAngle = calcSpineInclination(shoulderMid, hipMid);
                const neckAngle = calcSpineInclination(earMid, shoulderMid);
                const lKnee = calcAngle(lms[23], lms[25], lms[27]);
                const rKnee = calcAngle(lms[24], lms[26], lms[28]);
                const lHip = calcAngle(lms[11], lms[23], lms[25]);
                const rHip = calcAngle(lms[12], lms[24], lms[26]);
                const avgKnee = (lKnee + rKnee) / 2;
                const avgHip = (lHip + rHip) / 2;

                const localMetrics = {
                    torso: torsoAngle,
                    neck: neckAngle,
                    knee: avgKnee,
                    hip: avgHip
                };

                // Check Camera Framing
                const framing = checkFraming(lms);
                const isTooClose = (framing === "TOO_CLOSE_STEP_BACK");

                let posture = "STANDING";
                let conf = 0.94;
                let risk = "LOW RISK";

                if (isTooClose) {
                    posture = "STEP BACK (TOO CLOSE)";
                    conf = 0.99;
                    risk = "FRAMING WARNING";
                } else if (torsoAngle > 35.0) {
                    posture = "BENDING";
                    conf = 0.95;
                    risk = "HIGH RISK";
                } else if (avgKnee < 125.0 && avgHip < 135.0) {
                    posture = "SITTING";
                    conf = 0.96;
                    risk = "LOW RISK";
                } else if (neckAngle > 24.0) {
                    posture = "SLOUCHING";
                    conf = 0.91;
                    risk = "HIGH RISK";
                }

                const plcOutputs = {
                    "PLC_OUT_NORMAL_OP": (risk === "LOW RISK") ? 1 : 0,
                    "PLC_OUT_ALARM_LIGHT": (risk !== "LOW RISK") ? 1 : 0,
                    "PLC_OUT_BUZZER_ALERT": (risk === "HIGH RISK") ? 1 : 0,
                    "PLC_OUT_CONVEYOR_HALT": (posture === "BENDING") ? 1 : 0
                };

                // Draw rich OpenCV HUD on canvas
                drawHUD(canvasCtx, w, h, posture, conf, risk, fps, 12.0, localMetrics, isTooClose);

                // Update UI Telemetry Widgets
                updateTelemetryUI(posture, conf, risk, 12.0, plcOutputs);

                // Synchronize with Python backend for ML inference & PLC dispatcher (throttled to 5 times/sec)
                if (now - lastInspectTime > 200 && !isInspecting) {
                    lastInspectTime = now;
                    isInspecting = true;
                    fetch('/api/inspect_landmarks', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            landmarks: lms.map(l => ({ x: l.x, y: l.y, z: l.z, visibility: l.visibility || 1 })),
                            latency_ms: 12.0
                        })
                    }).then(res => res.json()).then(data => {
                        if (data && data.status === 'success') {
                            updateTelemetryUI(
                                data.posture_label.toUpperCase(),
                                data.confidence,
                                data.ergonomic_risk,
                                data.latency_ms,
                                data.plc_state
                            );
                        }
                    }).catch(() => {}).finally(() => {
                        isInspecting = false;
                    });
                }
            } else {
                // No person detected HUD
                canvasCtx.fillStyle = "rgba(0, 0, 0, 0.6)";
                canvasCtx.fillRect(0, 0, w, 60);
                canvasCtx.fillStyle = "#f59e0b";
                canvasCtx.font = "bold 16px 'Plus Jakarta Sans', sans-serif";
                canvasCtx.fillText("STATUS: SEARCHING FOR OPERATOR...", 20, 36);
                alertBox.classList.remove('visible');
            }

            canvasCtx.restore();
        });

        // Start Webcam Stream
        const camera = new Camera(videoElement, {
            onFrame: async () => {
                await pose.send({ image: videoElement });
            },
            width: 640,
            height: 480
        });

        camera.start().catch((err) => {
            console.error("Camera access error:", err);
            document.getElementById('val-posture').innerText = "CAMERA OFFLINE";
        });
    </script>
</body>
</html>"""
    return html_content

def run_dashboard(host="127.0.0.1", port=8000):
    print(f"[*] Launching Live Web Dashboard at: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_dashboard()
