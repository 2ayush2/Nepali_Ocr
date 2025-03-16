import cv2
import os
from utils import preprocess_document

def test_preprocess(image_path, output_image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}.")
        return

    # Preprocess to detect and crop the document (CamScanner-like)
    cropped_img = preprocess_document(img)
    cv2.imwrite(output_image_path, cropped_img)

    print(f"Preprocessed image saved to: {output_image_path}")

if __name__ == "__main__":
    input_dir = "input/"
    target_image = "sample1.jpg"  # Explicitly target sample2.jpg

    if os.path.exists(os.path.join(input_dir, target_image)) and target_image.endswith((".jpg", ".png")):
        image_path = os.path.join(input_dir, target_image)
        output_image_path = os.path.join(input_dir, f"preprocessed_{target_image}")
        test_preprocess(image_path, output_image_path)
    else:
        print(f"Error: {target_image} not found in {input_dir}. Please ensure the file exists.")