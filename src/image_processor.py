# src/image_processor.py
"""
Basic image processing utilities (filters, edge detection, saving).
"""

from pathlib import Path
import cv2
import numpy as np


def apply_gaussian_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to enhance contrast."""
    arr = (img * 255).astype("uint8")
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(arr)
    return out.astype(np.float32) / 255.0


def save_image(path: Path, img: np.ndarray) -> None:
    """Save image to disk as uint8 JPEG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = (img * 255).astype("uint8")
    cv2.imwrite(str(p), out)

if __name__ == "__main__":
    input_path = Path("sample_images/sample.jpeg")
    output_path = Path("output/filtered_sample.jpeg")

    print(f"🔍 Checking input path: {input_path.resolve()}")
    if not input_path.exists():
        print("❌ Input image not found. Please check the path and filename.")
        exit()

    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Failed to load image. cv2.imread returned None.")
        exit()

    print("✅ Image loaded. Shape:", img.shape)
    img = img.astype(np.float32) / 255.0

    enhanced = apply_clahe(img)
    blurred = apply_gaussian_blur(enhanced)

    save_image(output_path, blurred)
    print(f"✅ Filtered image saved to: {output_path.resolve()}")
