import mlflow
from ultralytics import YOLO

from ultralytics import settings

# Update a setting
settings.update({"mlflow": True})

# Reset settings to default values
settings.reset()


# Set MLflow tracking URI and token
mlflow.set_tracking_uri("https://dagshub.com/manna.phys/layout-segmentation-yolo.mlflow")
mlflow.set_experiment("YOLOv8 Experiment")


import dagshub
dagshub.init(repo_owner='manna.phys', repo_name='layout-segmentation-yolo', mlflow=True)


# Start an MLflow run
with mlflow.start_run():
    # Load and train your YOLOv8 model
    model = YOLO('yolov8n.pt')
    results = model.train(data='coco128.yaml', epochs=3)

    # Log parameters, metrics, and model
    mlflow.log_params({"epochs": 3, "model": "yolov8n.pt", "data": "coco128.yaml"})
    mlflow.log_metrics({"mAP_0.5": results.map50, "mAP_0.5:0.95": results.map})
    mlflow.pytorch.log_model(model, "model")

    # End the run
    mlflow.end_run()