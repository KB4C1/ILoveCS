from ultralytics import YOLO

model = YOLO("models/yolov8s.pt")  # базова модель
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="versions",
    name="v1"
)
