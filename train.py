from ultralytics import YOLO

def train_yolo():
    # Load YOLOv8m model (medium size for better accuracy)
    model = YOLO('yolov8m.pt')

    # Train the model
    model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='pan_exp',
        project='weights',
        exist_ok=True
    )

if __name__ == "__main__":
    train_yolo()