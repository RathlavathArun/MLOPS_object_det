from ultralytics import YOLO
import mlflow

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("yolo_project")

# Load base model
model = YOLO("yolov8n.pt")

with mlflow.start_run():

    # Train model
    results = model.train(
        data="data/data.yaml",
        epochs=30,
        imgsz=640
    )

    # Log parameters
    mlflow.log_param("model", "yolov8n")
    mlflow.log_param("epochs", 30)
    mlflow.log_param("imgsz", 640)

    # ✅ Get REAL metrics
    metrics = results.results_dict

    if "metrics/mAP50(B)" in metrics:
        mlflow.log_metric("mAP50", metrics["metrics/mAP50(B)"])

    if "metrics/precision(B)" in metrics:
        mlflow.log_metric("precision", metrics["metrics/precision(B)"])

    print("Training completed successfully!")