import os
import cv2

# ✅ Set your image path
image_path = "test-img/test/image/25_jpg.rf.2786fd23949c1bce66f93524f8a21d3e.jpg"
label_path = "test-img/test/label/25_jpg.rf.2786fd23949c1bce66f93524f8a21d3e.txt"

# ✅ Your class names (from data.yaml)
class_names = ["Name", "DOB", "Gender"]

# Load image
image = cv2.imread(image_path)
if image is None:
    print("❌ Could not read image.")
    exit()

h, w = image.shape[:2]

# Check label file
if not os.path.exists(label_path):
    print("❌ Label file not found.")
    exit()

with open(label_path, "r") as f:
    lines = f.readlines()

if not lines:
    print("⚠️ Label file is empty.")
    exit()

# Draw bounding boxes
for idx, line in enumerate(lines):
    try:
        parts = line.strip().split()
        cls_id, x, y, bw, bh = map(float, parts[:5])
        cls_id = int(cls_id)

        if cls_id >= len(class_names):
            print(f"❌ Invalid class ID: {cls_id}")
            continue

        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            print(f"⚠️ Bounding box out of image bounds on line {idx+1}")

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = class_names[cls_id]
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    except Exception as e:
        print(f"❌ Error reading line {idx+1}: {e}")

# Show image
cv2.imshow("Label Check", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
