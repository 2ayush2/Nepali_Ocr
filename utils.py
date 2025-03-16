import cv2
import numpy as np
import os
from glob import glob


def preprocess_image(image):
    """
    Preprocess image with robust contour detection
    Returns preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction with smaller kernel
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edge detection with adjusted thresholds
    edges = cv2.Canny(blurred, 20, 80, apertureSize=3)

    # Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=30, minLineLength=80, maxLineGap=10
    )
    print(f"Number of Hough lines detected: {len(lines) if lines is not None else 0}")

    if lines is not None:
        # Draw lines for contour detection
        line_image = np.zeros_like(image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Morphological operation to connect nearby lines
        kernel = np.ones((5, 5), np.uint8)
        line_image = cv2.dilate(line_image, kernel, iterations=2)

        gray_lines = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            gray_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        # Fallback to adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        # Morphological operation to connect regions
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

    print(f"Number of contours found: {len(contours)}")
    if contours:
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Try to find a quadrilateral
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if (
                len(approx) == 4 and cv2.contourArea(contour) > 100
            ):  # Further reduced area threshold
                box = approx
                print(
                    f"Quadrilateral contour found with area: {cv2.contourArea(contour)}"
                )
                break
        else:
            # Fallback: Use the largest contour's bounding rectangle
            contour = contours[0]
            x, y, w, h = cv2.boundingRect(contour)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            print(
                f"No quadrilateral found, using largest contour's bounding box with area: {cv2.contourArea(contour)}"
            )

        box = np.intp(box.reshape(4, 2))

        # Calculate perspective transform
        width = max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))
        height = max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))
        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )

        # Order points
        src_pts = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1)
        src_pts[0] = box[np.argmin(s)]  # Top-left
        src_pts[2] = box[np.argmax(s)]  # Bottom-right
        diff = np.diff(box, axis=1)
        src_pts[1] = box[np.argmin(diff)]  # Top-right
        src_pts[3] = box[np.argmax(diff)]  # Bottom-left

        # Apply perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))

        # Rotation detection
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(
            warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
        else:
            angle = 0  # Default to no rotation if no content is detected

        (h, w) = warped.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(warped, M, (w, h))

        # Enhance final image
        final = cv2.convertScaleAbs(rotated, alpha=1.5, beta=10)

        # Check if image is mostly black
        mean_val = cv2.mean(final)[0]
        if mean_val < 20:
            raise ValueError("Processed image is mostly black, likely a failure")

        return final
    else:
        # Ultimate fallback: Use the entire image with simple thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            final = image[y : y + h, x : x + w]
            print(
                f"Ultimate fallback: Using bounding box of largest thresholded region with area: {cv2.contourArea(contour)}"
            )
            return final
        else:
            raise ValueError("No significant region found in image even after fallback")


def process_images(input_folder="input", output_folder="preprocessed"):
    """
    Process all images in input folder and save to preprocessed folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = (
        glob(os.path.join(input_folder, "*.jpg"))
        + glob(os.path.join(input_folder, "*.png"))
        + glob(os.path.join(input_folder, "*.jpeg"))
    )

    if not image_files:
        print("No images found in input folder")
        return

    for image_path in image_files:
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load {image_path}")
                continue

            preprocessed = preprocess_image(img)
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder, f"processed_{filename}")
            cv2.imwrite(output_path, preprocessed)
            print(f"Saved preprocessed image to: {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")


def main():
    input_folder = "input"
    output_folder = "preprocessed"

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created input folder. Please place images in '{input_folder}'")
        return

    print(f"Processing images from '{input_folder}' to '{output_folder}'")
    process_images(input_folder, output_folder)


if __name__ == "__main__":
    main()
