import cv2
import numpy as np
import os


def preprocess_and_crop_image(
    input_image_path, output_cropped_path, output_enhanced_path
):
    """
    Preprocess the input image, crop the main region, and enhance it for OCR.
    """
    # Create output folders if they don't exist
    os.makedirs(os.path.dirname(output_cropped_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_enhanced_path), exist_ok=True)

    # 1. Load original image in color
    color_image = cv2.imread(input_image_path)
    if color_image is None:
        print(f"❌ Oops! Image not found at: {input_image_path}")
        return False
    print("🔹 Step 1: Original Color Image loaded")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    print("🔹 Step 2: Converted to Grayscale")

    # 3. Smooth and detect edges
    gray = cv2.GaussianBlur(gray, (47, 47), 0)
    edges = cv2.Canny(gray, 200, 200)
    print("🔹 Step 3: Canny Edge Detection")

    # 4. Otsu thresholding to get binary
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("🔹 Step 4: Binary Image using Otsu Thresholding")

    # 5. Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ Error: No contours found.")
        return False

    # 6. Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 7. Get rotated bounding box
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # np.int0 deprecated

    # 8. Mask and crop
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [box], 0, 255, -1)
    cropped = cv2.bitwise_and(color_image, color_image, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    cropped = cropped[y : y + h, x : x + w]

    # Save cropped image
    cv2.imwrite(output_cropped_path, cropped)
    print(f"✅ Cropped image saved to: {output_cropped_path}")

    # 9. Enhance the image
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(cv2.add(s, 50), 0, 255)
    v = np.clip(cv2.add(v, 50), 0, 255)
    enhanced = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)

    # Save enhanced image
    cv2.imwrite(output_enhanced_path, enhanced)
    print(f"✅ Enhanced image saved to: {output_enhanced_path}")

    return True


if __name__ == "__main__":
    input_path = "input/sample2.jpg"
    output_cropped_path = "preprocessed/output1_cropped.jpg"
    output_enhanced_path = "preprocessed/output2_enhanced.jpg"

    print(f"📁 Input Image Path: {input_path}")
    print(f"📁 Output Cropped Path: {output_cropped_path}")
    print(f"📁 Output Enhanced Path: {output_enhanced_path}")

    success = preprocess_and_crop_image(
        input_path, output_cropped_path, output_enhanced_path
    )

    if success:
        print("🎉 Preprocessing complete.")
    else:
        print("❌ Preprocessing failed.")
