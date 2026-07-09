import asyncio
import base64
import cv2
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import threading
import time
from collections import defaultdict

# ============================================================
# 🔧 FIX: Patch torch.load to use weights_only=False
# This must run BEFORE importing torch or ultralytics
# ============================================================
import torch

# Save original load function
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False for YOLO model loading
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = _patched_torch_load

# ============================================================
# Now import the rest normally
# ============================================================

app = FastAPI()

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model (now works with patched torch.load)
model = YOLO("yolov8n.pt")

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Detection stats
detection_stats = {
    "total_detections": 0,
    "class_counts": defaultdict(int),
    "recent_detections": []
}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print(f"Client connected. Total connections: {len(manager.active_connections)}")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                frame_data = payload.get("frame", "")
                if not frame_data:
                    continue

                img_bytes = base64.b64decode(frame_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                results = model(frame, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()

                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        detections.append({
                            "class": class_name,
                            "confidence": round(conf * 100, 1),
                            "class_id": cls
                        })
                        detection_stats["class_counts"][class_name] += 1
                        detection_stats["total_detections"] += 1

                detection_stats["recent_detections"] = detections[-20:]

                _, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

                response = {
                    "type": "detection",
                    "frame": jpeg_base64,
                    "detections": detections,
                    "count": len(detections),
                    "stats": {
                        "total": detection_stats["total_detections"],
                        "class_counts": dict(detection_stats["class_counts"])
                    },
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(response))

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client disconnected. Total connections: {len(manager.active_connections)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    return {"status": "ok", "connections": len(manager.active_connections)}

@app.get("/stats")
async def get_stats():
    return detection_stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
