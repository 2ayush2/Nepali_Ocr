# ocr.py
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import json
import os


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(binary)
    return denoised


def draw_boxes(image, boxes, texts, labels):
    # Create a copy of the image to draw on
    img = image.copy()
    for box, text, label in zip(boxes, texts, labels):
        x_min, y_min, x_max, y_max = map(int, box)
        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Put label and text
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
    return img


def run_ocr(image_path, weights_path, output_image_path, output_json_path):
    # Load YOLOv8m model and EasyOCR
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    reader = easyocr.Reader(["en"])

    # Read the preprocessed image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load preprocessed image from {image_path}")
        return

    # Detect text regions with YOLOv8m
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x_min, y_min, x_max, y_max]
    labels = [
        model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()
    ]  # Class labels

    # Extract text and prepare results
    texts = []
    result_data = []
    for box, label in zip(boxes, labels):
        # Skip non-text regions
        if label in ["EmblemLogo", "Goisybol", "QR Code"]:
            continue

        x_min, y_min, x_max, y_max = map(int, box)
        # Ensure ROI coordinates are within image bounds
        y_min, y_max = max(0, y_min), min(img.shape[0], y_max)
        x_min, x_max = max(0, x_min), min(img.shape[1], x_max)
        if x_max <= x_min or y_max <= y_min:
            continue  # Skip invalid ROIs

        roi = img[y_min:y_max, x_min:x_max]
        roi = preprocess_image(roi)
        ocr_result = reader.readtext(roi)
        text = ocr_result[0][1] if ocr_result else "No text detected"
        texts.append(text)

        # Store result for JSON
        result_data.append(
            {"box": [x_min, y_min, x_max, y_max], "label": label, "text": text}
        )

    # Annotate image with bounding boxes
    annotated_img = draw_boxes(img, boxes, texts, labels)
    cv2.imwrite(output_image_path, annotated_img)

    # Print results to console
    print("OCR Results for Aadhaar Card:")
    for i, data in enumerate(result_data):
        print(f"Region {i + 1}:")
        print(f"  Box: {data['box']}")
        print(f"  Label: {data['label']}")
        print(f"  Text: {data['text']}")

    # Save results to JSON file
    json_output = {"image_path": output_image_path, "results": result_data}
    with open(output_json_path, "w") as f:
        json.dump(json_output, f, indent=4)

    print(f"\nResults saved to JSON file: {output_json_path}")
    print(f"Annotated image saved to: {output_image_path}")


if __name__ == "__main__":
    input_path = "preprocessed/processed_with_boxes_sample2.jpg"
    weights_path = "weights/pan_exp/weights/best.pt"
    output_image_path = "output/annotated_sample2.jpg"
    output_json_path = "output/ocr_results_sample2.json"
    run_ocr(input_path, weights_path, output_image_path, output_json_path)
