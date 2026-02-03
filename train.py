from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("models/yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project="versions",
        name="v",
        workers=0,
        save_dir="runs/detect",
    )
