from ultralytics import YOLO

model = YOLO("runs/detect/aadhar-fields7/weights/best.pt")

results = model.predict(
    source="test-img/test/images/25_jpg.rf.2786fd23949c1bce66f93524f8a21d3e.jpg",
    conf=0.1,
)

for r in results:
    boxes = r.boxes
    for cls, conf in zip(boxes.cls, boxes.conf):
        print("Detected class:", int(cls), "Confidence:", float(conf))
