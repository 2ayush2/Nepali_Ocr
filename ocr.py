import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import easyocr
import torch


def draw_boxes(image, results, max_width=600):
    img = image.copy()
    for data in results:
        box = data["box"]
        label = data["label"]
        text = data["text"]
        if text.strip() in ["", "No text detected", "OCR failed"]:
            continue
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label_text = f"{label}: {text}"
        cv2.putText(
            img,
            label_text,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Resize image to max_width while keeping aspect ratio
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    return img


def clean_text(text):
    replacements = {"D0B": "DOB", "IND1A": "INDIA", "—": "-", "O": "0"}
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text.strip()


def is_valid_text(text):
    text = text.strip()
    if len(text) < 3:
        return False
    if all(c in "-–:." for c in text):  # symbols only
        return False
    return True


def run_ocr(image_path, weights_path, output_image_path, output_json_path):
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # Load YOLO model
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return

    # Initialize EasyOCR
    try:
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    except Exception as e:
        print(f"❌ Error initializing EasyOCR: {e}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return

    # YOLO inference
    try:
        results = model(img)
        if not results or len(results[0].boxes) == 0:
            print("⚠️ No regions detected.")
            return

        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        areas = [
            (box[2] - box[0]) * (box[3] - box[1]) for box in boxes
        ]  # area of each box
    except Exception as e:
        print(f"❌ YOLO detection failed: {e}")
        return

    skip_labels = ["Emblemlogo", "Goisymbol", "QR Code", "Seal", "Logo"]
    result_dict = {}  # key: label, value: (box, text)

    for i, (box, label, area) in enumerate(zip(boxes, labels, areas)):
        if label in skip_labels:
            continue

        x_min, y_min, x_max, y_max = map(int, box)
        y_min, y_max = max(0, y_min), min(img.shape[0], y_max)
        x_min, x_max = max(0, x_min), min(img.shape[1], x_max)

        if x_max <= x_min or y_max <= y_min:
            continue

        roi = img[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            continue

        try:
            ocr_result = reader.readtext(roi, detail=0)
            text = clean_text(ocr_result[0]) if ocr_result else "No text detected"
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            text = "OCR failed"

        if not is_valid_text(text):
            continue

        # If label exists already, keep the bigger region (area-based filtering)
        if label not in result_dict or area > result_dict[label]["area"]:
            result_dict[label] = {
                "box": [x_min, y_min, x_max, y_max],
                "label": label,
                "text": text,
                "area": area,
            }

    # Remove area field for final results
    final_results = []
    for data in result_dict.values():
        final_results.append(
            {"box": data["box"], "label": data["label"], "text": data["text"]}
        )

    # Draw and resize output image (600px max width)
    annotated_img = draw_boxes(img, final_results, max_width=600)
    cv2.imwrite(output_image_path, annotated_img)

    # Save JSON
    json_output = {"image_path": output_image_path, "results": final_results}
    with open(output_json_path, "w") as f:
        json.dump(json_output, f, indent=4)

    # Print output
    print("\n✅ FINAL OCR RESULTS:")
    for i, data in enumerate(final_results):
        print(f" {i+1}. [{data['label']}] → {data['text']}")

    print(f"\n📄 JSON saved: {output_json_path}")
    print(f"🖼️ Annotated image saved: {output_image_path}")


if __name__ == "__main__":
    # You can switch between cropped and enhanced images easily here
    # For accuracy, we're using the enhanced one
    input_path = "preprocessed/output2_enhanced.jpg"  # <-- best for OCR
    weights_path = "runs/detect/weights/aadhar-fields7/best.pt"
    output_image_path = "output/annotated_sample2.jpg"
    output_json_path = "output/ocr_results_sample2.json"

    run_ocr(input_path, weights_path, output_image_path, output_json_path)
