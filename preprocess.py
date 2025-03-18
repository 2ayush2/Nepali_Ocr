import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict


# Face Detection Abstract Base Class
class FaceDetector(ABC):
    @abstractmethod
    def changeOrientationUntilFaceFound(
        self, image: np.ndarray, rot_interval: int
    ) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def findFace(self, img: np.ndarray) -> float:
        pass

    @abstractmethod
    def rotate_bound(self, image: np.ndarray, angle: float) -> np.ndarray:
        pass


class SsdFaceDetector(FaceDetector):
    def __init__(self, confidence_threshold: float = 0.3, model_dir: str = "models"):
        self.model_dir = model_dir
        self.modelFile = os.path.join(
            model_dir, "res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.configFile = os.path.join(model_dir, "deploy.prototxt.txt")
        self.confidence_threshold = confidence_threshold

        if not self._verify_model_files():
            print("Model files missing or corrupted. Attempting to download...")
            self._download_models()

        self.net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)

    def _verify_model_files(self) -> bool:
        expected_model_size, expected_config_size = 2700000, 28000
        if not os.path.isfile(self.modelFile) or not os.path.isfile(self.configFile):
            return False
        return (
            os.path.getsize(self.modelFile) > expected_model_size * 0.9
            and os.path.getsize(self.configFile) > expected_config_size * 0.9
        )

    def _download_models(self) -> None:
        import urllib.request

        os.makedirs(self.model_dir, exist_ok=True)
        urls = {
            self.modelFile: "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            self.configFile: "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt",
        }
        for file_path, url in urls.items():
            urllib.request.urlretrieve(url, file_path)

    def changeOrientationUntilFaceFound(
        self, image: np.ndarray, rot_interval: int
    ) -> Optional[np.ndarray]:
        img = image.copy()
        face_conf = []
        for angle in range(0, 360, rot_interval):
            img_rotated = self.rotate_bound(img, angle)
            confidence = self.findFace(img_rotated)
            face_conf.append((confidence, angle))
        if not face_conf:
            return None
        max_conf_idx = np.argmax(np.array(face_conf)[:, 0])
        angle_max, max_confidence = (
            face_conf[max_conf_idx][1],
            face_conf[max_conf_idx][0],
        )
        return (
            self.rotate_bound(image, angle_max)
            if max_confidence > self.confidence_threshold
            else None
        )

    def findFace(self, img: np.ndarray) -> float:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
        )
        self.net.setInput(blob)
        faces = self.net.forward()
        return max([faces[0, 0, i, 2] for i in range(faces.shape[2])], default=0)

    def rotate_bound(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        cX, cY = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos_val, sin_val = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW, nH = int((h * sin_val) + (w * cos_val)), int((h * cos_val) + (w * sin_val))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))


class FaceFactory(ABC):
    @abstractmethod
    def get_face_detector(self) -> FaceDetector:
        pass


class SsdModel(FaceFactory):
    def get_face_detector(self) -> FaceDetector:
        return SsdFaceDetector()


def face_factory(face_model: str = "ssd") -> FaceFactory:
    return {"ssd": SsdModel()}[face_model]


# Placeholder for advanced segmentation (replace with DeepLabv3 or similar in production)
class UnetModel:
    def __init__(self, device: str):
        self.device = device

    def predict(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        return mask


# Image Analysis and Processing Functions
def analyze_image(img: np.ndarray) -> Dict[str, bool]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = np.mean(gray)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    tilt_angle = (
        np.mean([(theta * 180 / np.pi) - 90 for rho, theta in lines[:, 0]])
        if lines is not None
        else 0
    )
    background_present = (
        np.mean(cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]) > 10
    )
    return {
        "is_blurry": laplacian_var < 100,
        "poor_lighting": mean_brightness < 50 or mean_brightness > 200,
        "tilt_angle": abs(tilt_angle),
        "background_present": background_present,
        "hand_obstruction": detect_hand(img),
        "is_clear": laplacian_var >= 100
        and 50 <= mean_brightness <= 200
        and abs(tilt_angle) < 5
        and not background_present
        and not detect_hand(img),
    }


def detect_hand(img: np.ndarray) -> bool:
    # Simple heuristic: detect skin-like colors in HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype="uint8")
    upper_skin = np.array([20, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return (
        cv2.contourArea(
            cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        )
        > 1000
        if cv2.countNonZero(mask) > 0
        else False
    )


def remove_background_and_hands(img: np.ndarray, model: UnetModel) -> np.ndarray:
    mask = model.predict(img)
    # Enhance mask to remove hand obstructions
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    return cv2.bitwise_and(img, img, mask=mask)


def correct_perspective(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) != 4:
        return img
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    width = int(
        max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
    )
    height = int(
        max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))
    )
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (width, height))


def enhance_image(img: np.ndarray, analysis: Dict[str, bool]) -> np.ndarray:
    if analysis["is_blurry"]:
        img = cv2.addWeighted(
            img, 1.5, cv2.GaussianBlur(img, (0, 0), 10), -0.5, 0
        )  # Deblur
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # Noise reduction
    if analysis["poor_lighting"]:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=3.0 if np.mean(l) < 75 else 1.5, tileGridSize=(8, 8)
        )
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    h, w = img.shape[:2]
    if h * w < 500000:  # Super-resolution for low-res images
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return img


def smart_crop(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    if not (1.4 <= aspect_ratio <= 1.6):  # Validate ID card aspect ratio
        return img
    return img[y : y + h, x : x + w]


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_val, sin_val = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int((h * sin_val) + (w * cos_val)), int((h * cos_val) + (w * sin_val))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(img, M, (nW, nH))


def find_orientation(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return 0
    angles = [(theta * 180 / np.pi) - 90 for rho, theta in lines[:, 0]]
    hist, bins = np.histogram(angles, bins=36, range=(-90, 90))
    return ((bins[np.argmax(hist)] + 90) % 180) - 90


def process_id_card(
    input_path: str, output_dir: str, debug: bool = False
) -> Optional[str]:
    use_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    face_detector = face_factory().get_face_detector()
    model = UnetModel(use_cuda)

    if not os.path.isfile(input_path):
        print(f"Error: Input image not found at {input_path}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to load {input_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: Analyze image
    analysis = analyze_image(img_rgb)
    print(f"Image analysis: {analysis}")
    if analysis["is_clear"]:
        print("Image is already perfect. Returning as is.")
        output_filename = os.path.join(
            output_dir, f"processed_{os.path.basename(input_path)}"
        )
        cv2.imwrite(output_filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return output_filename

    # Step 2: Face detection and initial orientation
    aligned_img = face_detector.changeOrientationUntilFaceFound(img_rgb, 10)
    if aligned_img is None:
        print("No face detected; proceeding with basic alignment.")
        aligned_img = img_rgb

    # Step 3: Background and hand removal
    if analysis["background_present"] or analysis["hand_obstruction"]:
        aligned_img = remove_background_and_hands(aligned_img, model)

    # Step 4: Perspective correction
    corrected_img = correct_perspective(aligned_img)

    # Step 5: Fine-tuned orientation correction
    orientation_angle = find_orientation(corrected_img)
    if abs(orientation_angle) > 1:
        corrected_img = rotate_image(corrected_img, orientation_angle)
        print(f"Rotated image by {orientation_angle:.2f} degrees.")

    # Step 6: AI-based enhancement
    enhanced_img = enhance_image(corrected_img, analysis)

    # Step 7: Intelligent cropping
    final_img = smart_crop(enhanced_img)

    # Step 8: Debugging and output
    if debug:
        plt.subplot(221), plt.imshow(img_rgb), plt.title("Original")
        plt.subplot(222), plt.imshow(aligned_img), plt.title("Aligned")
        plt.subplot(223), plt.imshow(corrected_img), plt.title("Corrected")
        plt.subplot(224), plt.imshow(final_img), plt.title("Final")
        plt.show()

    output_filename = os.path.join(
        output_dir, f"processed_{os.path.basename(input_path)}"
    )
    cv2.imwrite(output_filename, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"Saved processed ID card to: {output_filename}")
    return output_filename


if __name__ == "__main__":
    img_path = "input/sample2.jpg"
    preprocess_dir = "output"
    processed_image_path = process_id_card(img_path, preprocess_dir, debug=True)
