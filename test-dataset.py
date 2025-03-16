import cv2
import os
import glob

# Define dataset directory
dataset_path = "Test_adhar/train/"
image_dir = os.path.join(dataset_path, "images")
label_dir = os.path.join(dataset_path, "labels")

# Class mapping from data.yaml
class_names = {
    0: "Dob",
    1: "Gender",
    2: "Name",
    3: "Aadhaar Number",
}

# Fetch all image files
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))

# Process each image
for image_path in image_files:
    # Construct label file path
    base_name = os.path.basename(image_path).replace(".jpg", ".txt")
    label_path = os.path.join(label_dir, base_name)

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image {image_path}")
        continue

    h, w, _ = img.shape  # Get image dimensions

    # Check if label file exists
    if not os.path.exists(label_path):
        print(f"⚠️ Warning: Label file not found: {label_path}")
        continue

    # Read the YOLO label file
    with open(label_path, "r") as file:
        labels = file.readlines()

    bbox_found = False  # Track if a bounding box is drawn

    # Process each bounding box
    for idx, label in enumerate(labels):
        label_data = label.strip().split()

        if len(label_data) < 5:
            print(f"⚠️ Skipping invalid label: {label_data}")
            continue

        try:
            class_id = int(label_data[0])
            x_center, y_center, width, height = map(float, label_data[1:5])

            # Convert normalized coordinates to pixel values
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # Ensure bounding box is within image boundaries
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{idx+1}. {class_names.get(class_id, 'Unknown')}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            bbox_found = True

        except ValueError:
            print(f"⚠️ Skipping corrupted label data: {label_data}")

    # If no valid bounding box was found
    if not bbox_found:
        print(f"⚠️ No valid bounding boxes found for {image_path}")

    # Show labeled image
    cv2.imshow("Labeled Image", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
