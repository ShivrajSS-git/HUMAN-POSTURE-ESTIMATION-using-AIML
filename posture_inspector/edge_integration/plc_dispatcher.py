import json
import time

class IndustrialPLCDispatcher:
    """
    Industrial Edge & PLC (Programmable Logic Controller) Signal Telemetry Dispatcher.
    Simulates digital I/O pin triggers, conveyor safety interlocks, and MQTT/JSON telemetry payloads.
    """
    def __init__(self, publish_mqtt=False):
        self.publish_mqtt = publish_mqtt
        self.total_inspections = 0
        self.high_risk_count = 0
        self.last_state = None

    def dispatch_signal(self, posture_label, confidence, ergonomic_risk, latency_ms):
        """
        Process current frame state and determine digital I/O signal states for PLC interlock.
        
        Returns:
            dict of PLC channel states and telemetry payload
        """
        self.total_inspections += 1

        is_high_risk = (ergonomic_risk == "HIGH RISK") or (posture_label in ["slouching", "bending"])
        if is_high_risk:
            self.high_risk_count += 1

        # Digital I/O Output Channel Simulation
        plc_channels = {
            "PLC_OUT_NORMAL_OP": 0 if is_high_risk else 1,
            "PLC_OUT_ALARM_LIGHT": 1 if is_high_risk else 0,
            "PLC_OUT_BUZZER_ALERT": 1 if (is_high_risk and confidence > 0.85) else 0,
            "PLC_OUT_CONVEYOR_HALT": 1 if (posture_label == "bending" and confidence > 0.90) else 0,
        }

        telemetry_payload = {
            "timestamp": time.time(),
            "operator_posture": posture_label,
            "model_confidence": round(float(confidence), 3),
            "ergonomic_risk": ergonomic_risk,
            "inference_latency_ms": round(float(latency_ms), 2),
            "plc_digital_outputs": plc_channels,
            "telemetry_stats": {
                "total_frames_inspected": self.total_inspections,
                "high_risk_frames": self.high_risk_count,
                "risk_ratio": round(self.high_risk_count / max(1, self.total_inspections), 3)
            }
        }

        self.last_state = telemetry_payload
        return telemetry_payload

    def get_json_telemetry(self):
        """Get formatted JSON telemetry string for MQTT/REST transmission."""
        if self.last_state:
            return json.dumps(self.last_state, indent=2)
        return json.dumps({"status": "OFFLINE"})
