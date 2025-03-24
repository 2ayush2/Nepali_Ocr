from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="test-img/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    name="aadhar-fields7",
    cache=False,
    verbose=True,
)
