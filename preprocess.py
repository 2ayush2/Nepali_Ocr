# preprocess.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import time
from abc import ABC, abstractmethod
from math import atan2, cos, sin, sqrt, pi


# Face Detection Abstract Base Class
class FaceDetector(ABC):
    @abstractmethod
    def changeOrientationUntilFaceFound(self, image, rot_interval):
        pass

    @abstractmethod
    def findFace(self, img):
        pass

    @abstractmethod
    def rotate_bound(self, image, angle):
        pass


# SSD Face Detector Implementation
class SsdFaceDetector(FaceDetector):
    def __init__(self, confidence_threshold=0.3):
        self.model_dir = "models"
        self.modelFile = os.path.join(
            self.model_dir, "res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.configFile = os.path.join(self.model_dir, "deploy.prototxt.txt")
        self.confidence_threshold = confidence_threshold

        # Check if model files exist
        model_file_exists = os.path.isfile(self.modelFile)
        config_file_exists = os.path.isfile(self.configFile)
        print(f"Checking model files in {self.model_dir}:")
        print(f"  - {self.modelFile} exists: {model_file_exists}")
        print(f"  - {self.configFile} exists: {config_file_exists}")

        if not model_file_exists or not config_file_exists:
            raise FileNotFoundError(f"Model files not found in {self.model_dir}")

    def changeOrientationUntilFaceFound(self, image, rot_interval):
        img = image.copy()
        face_conf = []
        for angle in range(0, 360, rot_interval):
            img_rotated = self.rotate_bound(img, angle)
            confidence = self.findFace(img_rotated)
            face_conf.append((confidence, angle))
            print(f"Angle: {angle}°, Confidence: {confidence:.2f}")
        face_confidence = np.array(face_conf)
        face_arg_max = np.argmax(face_confidence, axis=0)
        angle_max = face_confidence[face_arg_max[0]][1]
        max_confidence = face_confidence[face_arg_max[0]][0]
        print(f"Max confidence: {max_confidence:.2f} at angle: {angle_max}°")
        rotated_img = self.rotate_bound(image, angle_max)
        return rotated_img if max_confidence > self.confidence_threshold else None

    def findFace(self, img):
        FaceNet = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
        )
        FaceNet.setInput(blob)
        faces = FaceNet.forward()
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0:
                print(
                    f"Face detected with confidence: {confidence:.2f} at index {i} (x={faces[0, 0, i, 3]*w:.0f}, y={faces[0, 0, i, 4]*h:.0f}, w={faces[0, 0, i, 5]*w-faces[0, 0, i, 3]*w:.0f}, h={faces[0, 0, i, 6]*h-faces[0, 0, i, 4]*h:.0f})"
                )
                return confidence
        print("No face detected with confidence above 0")
        return 0

    def rotate_bound(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1])
        nW = int((h * sin_val) + (w * cos_val))
        nH = int((h * cos_val) + (w * sin_val))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))


# Face Factory
class FaceFactory(ABC):
    @abstractmethod
    def get_face_detector(self) -> FaceDetector:
        pass


class SsdModel(FaceFactory):
    def get_face_detector(self) -> FaceDetector:
        return SsdFaceDetector(confidence_threshold=0.3)


def face_factory(face_model="ssd", confidence_threshold=0.3) -> FaceFactory:
    factories = {"ssd": SsdModel()}
    return factories[face_model]


# Utility Functions
class UnetModel:
    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device

    def predict(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask


class Res34BackBone:
    pass  # Placeholder for U-Net backbone


def correctPerspective(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)
    _, thresh = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    cntrs, _ = cv2.findContours(img_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_max = max(cntrs, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt_max, 0.02 * cv2.arcLength(cnt_max, True), True)
    (height_q, width_q) = img.shape[:2]
    return warpImg(img, approx, width_q, height_q)


def reorder(myPoints):
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def warpImg(img, points, w, h):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h))


def findOrientationofLines(mask):
    # Find contours in the mask
    cntrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cntrs) == 0:
        return 0

    # Filter contours to focus on text regions (exclude very large or very small contours)
    filtered_cntrs = [
        cnt
        for cnt in cntrs
        if 500 < cv2.contourArea(cnt) < 0.5 * mask.shape[0] * mask.shape[1]
    ]
    if not filtered_cntrs:
        return 0

    # Use Hough Line Transform to detect lines in the mask
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        # Fallback to PCA if Hough Lines fails
        largest_cnt = max(filtered_cntrs, key=cv2.contourArea)
        return getOrientation(largest_cnt, mask)

    # Compute the average angle of detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90  # Convert to degrees and adjust
        angles.append(angle)

    if not angles:
        return 0

    avg_angle = np.mean(angles)
    # Normalize the angle to be between -90 and 90 degrees
    avg_angle = ((avg_angle + 90) % 180) - 90
    return avg_angle


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    mean, eigenvectors, _ = cv2.PCACompute2(data_pts, np.empty((0)))
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    return np.rad2deg(angle)


def rotateImage(orientation_angle, final_img):
    (h, w) = final_img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), orientation_angle, 1.0)
    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])
    nW = int((h * sin_val) + (w * cos_val))
    nH = int((h * cos_val) + (w * sin_val))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(final_img, M, (nW, nH))


def displayAllBoxes(img, rect):
    for rct in rect:
        x1, w, y1, h = rct
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 1)
        cX = round(int(x1) + w / 2.0)
        cY = round(int(y1) + h / 2.0)
        cv2.circle(img, (cX, cY), 3, (0, 255, 0), -1)
    return img


def displayMachedBoxes(img, new_bboxes):
    for box in new_bboxes:
        x1, w, y1, h = box
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 3)
        cX = round(int(x1) + w / 2.0)
        cY = round(int(y1) + h / 2.0)
        cv2.circle(img, (cX, cY), 7, (0, 255, 255), -1)
    return img


def createHeatMapAndBoxCoordinates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cntrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in cntrs if cv2.contourArea(cnt) > 100]
    return thresh, [(x, y, x + w, y, x, y + h, x + w, y + h) for x, y, w, h in boxes]


def getCenterRatios(img, centers):
    if len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        img_h, img_w, _ = img.shape
    ratios = np.zeros_like(centers, dtype=np.float32)
    for i, center in enumerate(centers):
        ratios[i] = (center[0] / img_w, center[1] / img_h)
    return ratios


def matchCenters(ratios1, ratios2):
    if len(ratios1) != 4:
        raise ValueError(f"Expected 4 centers in ratios1, got {len(ratios1)}")
    bbb0 = np.zeros_like(ratios2)
    bbb1 = np.zeros_like(ratios2)
    bbb2 = np.zeros_like(ratios2)
    bbb3 = np.zeros_like(ratios2)

    for i, r2 in enumerate(ratios2):
        bbb0[i] = abs(ratios1[0] - r2)
        bbb1[i] = abs(ratios1[1] - r2)
        bbb2[i] = abs(ratios1[2] - r2)
        bbb3[i] = abs(ratios1[3] - r2)

    sum_b0 = np.sum(bbb0, axis=1)
    sum_b1 = np.sum(bbb1, axis=1)
    sum_b2 = np.sum(bbb2, axis=1)
    sum_b3 = np.sum(bbb3, axis=1)

    arg_min_b0 = np.argmin(sum_b0)
    arg_min_b1 = np.argmin(sum_b1)
    arg_min_b2 = np.argmin(sum_b2)
    arg_min_b3 = np.argmin(sum_b3)

    return np.array([arg_min_b0, arg_min_b1, arg_min_b2, arg_min_b3])


def getCenterOfMasks(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    if len(contours) < 4:
        # Pad with dummy centers if fewer than 4 contours are found
        detected_centers = [(0, 0)] * (4 - len(contours))
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cX = round(int(x) + w / 2.0)
            cY = round(int(y) + h / 2.0)
            detected_centers.append((cX, cY))
    else:
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (cnts, boundingBoxes) = zip(
            *sorted(zip(contours, boundingBoxes), key=lambda b: b[1][1], reverse=False)
        )
        detected_centers = []
        for contour in cnts:
            (x, y, w, h) = cv2.boundingRect(contour)
            cX = round(int(x) + w / 2.0)
            cY = round(int(y) + h / 2.0)
            detected_centers.append((cX, cY))
    return np.array(detected_centers)


def getBoxRegions(regions):
    boxes = []
    centers = []
    for box_region in regions:
        # Unpack tuple directly instead of reshaping
        x1, y1, x2, y2, x3, y3, x4, y4 = box_region  # Already a tuple of 8 values
        x = min(x1, x3)
        y = min(y1, y2)
        w = abs(min(x1, x3) - max(x2, x4))
        h = abs(min(y1, y2) - max(y3, y4))
        cX = round(int(x) + w / 2.0)
        cY = round(int(y) + h / 2.0)
        centers.append((cX, cY))
        bbox = (int(x), w, int(y), h)
        boxes.append(bbox)
    return np.array(boxes), np.array(centers)


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


def run_preprocess(input_path, output_dir):
    # Configuration
    rotation_interval = 10
    ORI_THRESH = 1
    use_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold = 0.3

    # Check if input image exists
    if not os.path.isfile(input_path):
        print(f"Error: Input image not found at {input_path}")
        return None

    # Initialize components
    face_detector = face_factory(
        face_model="ssd", confidence_threshold=confidence_threshold
    ).get_face_detector()
    model = UnetModel(Res34BackBone(), use_cuda)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at: {output_dir}")

    # Read image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to load {input_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show original image
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(f"Original Image: {os.path.basename(input_path)}")
    plt.axis("off")
    plt.show()

    # Step 1: Face Detection and Orientation
    final_img = face_detector.changeOrientationUntilFaceFound(
        img_rgb, rotation_interval
    )
    if final_img is None:
        print(f"No face detected in identity card {os.path.basename(input_path)}")
        return None

    # Step 2: Perspective Correction
    final_img = correctPerspective(final_img)

    # Step 3: Segmentation and Orientation Correction
    txt_heat_map, regions = createHeatMapAndBoxCoordinates(final_img)
    txt_heat_map = cv2.cvtColor(txt_heat_map, cv2.COLOR_BGR2RGB)
    predicted_mask = model.predict(txt_heat_map)

    # Show predicted mask
    plt.figure(figsize=(8, 6))
    plt.imshow(predicted_mask, cmap="gray")
    plt.title(f"Predicted Mask for {os.path.basename(input_path)}")
    plt.axis("off")
    plt.show()

    # Track cumulative rotation
    total_rotation = 0

    # Automatic orientation correction
    orientation_angle = findOrientationofLines(predicted_mask.copy())
    print(f"Computed Orientation of ID Card is {orientation_angle:.2f} degrees")
    if abs(orientation_angle) > ORI_THRESH:
        print(
            f"Absolute orientation angle ({abs(orientation_angle):.2f}) is greater than ORI_THRESH ({ORI_THRESH})"
        )
        final_img = rotateImage(orientation_angle, final_img)
        total_rotation += orientation_angle
        print(f"Total rotation applied: {total_rotation:.2f} degrees")
        txt_heat_map, regions = createHeatMapAndBoxCoordinates(final_img)
        txt_heat_map = cv2.cvtColor(txt_heat_map, cv2.COLOR_BGR2RGB)
        predicted_mask = model.predict(txt_heat_map)
        # Show updated mask and image after automatic rotation
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(final_img)
        plt.title(f"Image After Automatic Rotation for {os.path.basename(input_path)}")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap="gray")
        plt.title(
            f"Updated Mask After Automatic Rotation for {os.path.basename(input_path)}"
        )
        plt.axis("off")
        plt.show()

    # Step 4: Detect and Match Bounding Boxes
    bbox_coordinates, box_centers = getBoxRegions(regions)
    mask_centers = getCenterOfMasks(predicted_mask)
    centers_ratio_mask = getCenterRatios(predicted_mask, mask_centers)
    centers_ratio_all = getCenterRatios(final_img, box_centers)
    matched_box_indexes = matchCenters(centers_ratio_mask, centers_ratio_all)
    new_bboxes = (
        bbox_coordinates[matched_box_indexes]
        if len(bbox_coordinates) >= 4
        else bbox_coordinates
    )

    # Step 5: Save Preprocessed Image
    final_img_with_matched_boxes = displayMachedBoxes(final_img.copy(), new_bboxes)
    final_img_with_all_boxes = displayAllBoxes(
        final_img_with_matched_boxes.copy(), bbox_coordinates
    )
    output_filename = os.path.join(
        output_dir, f"processed_{os.path.basename(input_path)}"
    )
    final_img_with_all_boxes_bgr = cv2.cvtColor(
        final_img_with_all_boxes, cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(output_filename, final_img_with_all_boxes_bgr)
    print(f"Saved preprocessed image to: {output_filename}")

    return output_filename


if __name__ == "__main__":
    input_path = "input/sample2.jpg"
    output_dir = "output"
    preprocess_image_path = run_preprocess(input_path, output_dir)
