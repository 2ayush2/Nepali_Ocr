import os
import cv2
import numpy as np
from easyocr import Reader
import matplotlib.pyplot as plt

# Configuration
img_dir = "test-img/train/images"
label_dir = "test-img/train/labels"
classes = ["Name", "DOB", "Gender"]
reader = Reader(["en"], gpu=False)


def yolo_to_pixel(x, y, w, h, img_w, img_h):
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2


# Loop through images
for img_name in os.listdir(img_dir):
    if not img_name.endswith((".jpg", ".png")):
        continue
    image_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"⚠️ Missing label for {img_name}")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    print(f"\n🔍 Image: {img_name}")
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x, y, bw, bh = map(float, parts[1:5])
        x1, y1, x2, y2 = yolo_to_pixel(x, y, bw, bh, w, h)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ocr_result = reader.readtext(crop)
        text = ocr_result[0][1] if ocr_result else "N/A"

        print(f"📦 Class: {classes[class_id]}")
        print(f"📝 OCR Text: {text}")
        print(
            f"{'✅ Match' if classes[class_id].lower() in text.lower() else '❌ Mismatch'}"
        )
