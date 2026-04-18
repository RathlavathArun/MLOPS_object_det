from prometheus_client import Counter, generate_latest, Gauge
from fastapi.responses import Response, StreamingResponse
from fastapi import FastAPI, UploadFile, File, Form, Query
from ultralytics import YOLO
import cv2
from collections import Counter as CCounter

app = FastAPI()

# Load trained model
model = YOLO("yolov8n.pt")
# model = YOLO("models/best.pt")

# Prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests")
DATA_DRIFT = Gauge("data_drift_score", "Data Drift Score")
MODEL_ACCURACY = Gauge("model_accuracy", "Model Accuracy")
@app.get("/")
def home():
    return {"message": "API Running"}

# ---------------- VIDEO UPLOAD ----------------
@app.post("/upload_video")
async def upload_video(
    file: UploadFile = File(...),
    target: str = Form(""),
    conf_threshold: float = Form(0.5)
):
    REQUEST_COUNT.inc()

    input_path = "input.mp4"
    output_path = "output.mp4"

    # Save video
    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        20,
        (640, 480)
    )

    detected_classes = []   # 🔥 FOR DRIFT

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, conf=conf_threshold, iou=0.4, imgsz=640)[0]

        h, w, _ = frame.shape

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()

            detected_classes.append(label)   # 🔥 TRACK

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Remove huge boxes
            if (x2 - x1) * (y2 - y1) > 0.6 * (w * h):
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # 🔥 CALCULATE DRIFT
    if detected_classes:
        counts = CCounter(detected_classes)

        total = sum(counts.values()) + 1e-6

# take top 2 classes dynamically
        values = list(counts.values())

        if len(values) >= 2:
            values.sort(reverse=True)
            drift_score = abs((values[0]/total) - (values[1]/total))
        else:
            drift_score = 0.0

        DATA_DRIFT.set(drift_score)

    return {"status": "done", "video": output_path}


# ---------------- LIVE STREAM ----------------
def generate_frames(target=None, conf_threshold=0.5):
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, conf=conf_threshold, iou=0.4, imgsz=640)[0]

        h, w, _ = frame.shape
        detected_classes = []   # 🔥 PER FRAME

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()

            detected_classes.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x2 - x1) * (y2 - y1) > 0.6 * (w * h):
                continue

            if target and target.strip().lower() != label:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            

            
       # MODEL_ACCURACY.set(0.6, 0.9)


        # 🔥 UPDATE DRIFT LIVE
        if detected_classes:
            counts = CCounter(detected_classes)

            car = counts.get("car", 0)
            person = counts.get("person", 0)

            total = car + person + 1e-6

            drift_score = abs((car / total) - (person / total))

            DATA_DRIFT.set(drift_score)

        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@app.get("/detect_video")
def detect_video(target: str = Query(None), conf_threshold: float = Query(0.5)):
    return StreamingResponse(
        generate_frames(target, conf_threshold),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")