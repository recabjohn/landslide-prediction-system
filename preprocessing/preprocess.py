"""
Image Preprocessing Module
===========================
Prepares satellite images for YOLOv8 inference by performing:
  - Resizing to model input dimensions (640×640)
  - Normalization of pixel values
  - Color space conversion (BGR ↔ RGB)
  - Contrast enhancement for satellite imagery
  - Format adjustment for batch inference
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocesses satellite images for YOLO object detection inference.

    Handles all necessary transformations to convert raw satellite imagery
    into the format expected by YOLOv8 models.
    """

    # Default YOLO input size
    DEFAULT_SIZE = 640

    def __init__(self, target_size=DEFAULT_SIZE):
        """
        Initialize the preprocessor.

        Args:
            target_size (int): Target image dimension (square). Default: 640
        """
        self.target_size = target_size

    # ----------------------------------------------------------------
    # CORE OPERATIONS
    # ----------------------------------------------------------------

    def resize(self, image, size=None, keep_aspect=False):
        """
        Resize image to target dimensions.

        Args:
            image (np.ndarray): Input image (H, W, C)
            size (int, optional): Target size. Uses self.target_size if None
            keep_aspect (bool): If True, pad to maintain aspect ratio

        Returns:
            np.ndarray: Resized image (size, size, C)
        """
        target = size or self.target_size

        if keep_aspect:
            return self._resize_with_padding(image, target)

        resized = cv2.resize(image, (target, target), interpolation=cv2.INTER_LINEAR)
        logger.debug(f"Resized from {image.shape[:2]} to ({target}, {target})")
        return resized

    def normalize(self, image, method="minmax"):
        """
        Normalize pixel values.

        Args:
            image (np.ndarray): Input image (uint8, 0-255)
            method (str): Normalization method
                - 'minmax': Scale to [0, 1]
                - 'standard': Zero mean, unit variance
                - 'imagenet': ImageNet mean/std normalization

        Returns:
            np.ndarray: Normalized image (float32)
        """
        img_float = image.astype(np.float32)

        if method == "minmax":
            # Scale to [0, 1]
            img_min, img_max = img_float.min(), img_float.max()
            if img_max - img_min > 0:
                img_float = (img_float - img_min) / (img_max - img_min)
            else:
                img_float = np.zeros_like(img_float)

        elif method == "standard":
            # Zero mean, unit variance per channel
            for c in range(img_float.shape[2]):
                mean = img_float[:, :, c].mean()
                std = img_float[:, :, c].std()
                if std > 0:
                    img_float[:, :, c] = (img_float[:, :, c] - mean) / std

        elif method == "imagenet":
            # ImageNet normalization (RGB order)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_float = img_float / 255.0
            img_float = (img_float - mean) / std

        logger.debug(f"Normalized with method='{method}'")
        return img_float

    def convert_color(self, image, conversion="bgr2rgb"):
        """
        Convert image color space.

        Args:
            image (np.ndarray): Input image
            conversion (str): Color conversion type
                - 'bgr2rgb' or 'rgb2bgr'
                - 'bgr2gray' or 'rgb2gray'
                - 'bgr2hsv' or 'rgb2hsv'

        Returns:
            np.ndarray: Color-converted image
        """
        conversions = {
            "bgr2rgb": cv2.COLOR_BGR2RGB,
            "rgb2bgr": cv2.COLOR_RGB2BGR,
            "bgr2gray": cv2.COLOR_BGR2GRAY,
            "rgb2gray": cv2.COLOR_RGB2GRAY,
            "bgr2hsv": cv2.COLOR_BGR2HSV,
            "rgb2hsv": cv2.COLOR_RGB2HSV,
        }

        if conversion not in conversions:
            raise ValueError(f"Unknown conversion '{conversion}'. Options: {list(conversions.keys())}")

        converted = cv2.cvtColor(image, conversions[conversion])
        logger.debug(f"Color conversion: {conversion}")
        return converted

    def enhance_contrast(self, image, method="clahe", clip_limit=2.0, grid_size=(8, 8)):
        """
        Enhance image contrast — useful for satellite images with low dynamic range.

        Args:
            image (np.ndarray): Input BGR image (uint8)
            method (str): Enhancement method ('clahe', 'histogram', 'auto')
            clip_limit (float): CLAHE clip limit
            grid_size (tuple): CLAHE tile grid size

        Returns:
            np.ndarray: Contrast-enhanced image (uint8)
        """
        if method == "clahe":
            # Apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l_enhanced = clahe.apply(l_channel)

            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        elif method == "histogram":
            # Simple histogram equalization per channel
            channels = cv2.split(image)
            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
            result = cv2.merge(eq_channels)

        elif method == "auto":
            # Auto brightness/contrast adjustment
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)

            alpha = 1.5 if mean_val < 100 else (1.2 if mean_val < 150 else 1.0)
            beta = 30 if mean_val < 100 else (10 if mean_val < 150 else 0)

            result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        else:
            raise ValueError(f"Unknown method '{method}'. Options: 'clahe', 'histogram', 'auto'")

        logger.debug(f"Contrast enhanced with method='{method}'")
        return result

    # ----------------------------------------------------------------
    # FULL PIPELINE
    # ----------------------------------------------------------------

    def prepare_for_inference(self, image, enhance=False):
        """
        Complete preprocessing pipeline: resize → enhance (optional) → prepare.

        This is the main entry point for preparing images for YOLO inference.
        YOLO handles its own normalization internally, so we only resize and
        optionally enhance contrast.

        Args:
            image (np.ndarray): Input BGR image (uint8)
            enhance (bool): Apply CLAHE contrast enhancement

        Returns:
            np.ndarray: Preprocessed image ready for YOLO inference (uint8, H×W×3)
        """
        # Step 1: Resize to target dimensions
        processed = self.resize(image, self.target_size)

        # Step 2: Optional contrast enhancement
        if enhance:
            processed = self.enhance_contrast(processed, method="clahe")

        # Ensure uint8 output (YOLO expects 0-255 BGR/RGB)
        if processed.dtype != np.uint8:
            processed = np.clip(processed * 255, 0, 255).astype(np.uint8)

        logger.info(f"Image prepared for inference: shape={processed.shape}, dtype={processed.dtype}")
        return processed

    # ----------------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------------

    def _resize_with_padding(self, image, target_size):
        """
        Resize while maintaining aspect ratio, padding the remainder.

        Args:
            image (np.ndarray): Input image
            target_size (int): Target dimension

        Returns:
            np.ndarray: Padded image (target_size, target_size, C)
        """
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas (dark grey background)
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

        # Center the resized image
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        logger.debug(f"Resized with padding: {image.shape[:2]} → ({target_size}, {target_size})")
        return canvas

    @staticmethod
    def get_image_info(image):
        """
        Get image metadata.

        Args:
            image (np.ndarray): Input image

        Returns:
            dict: Image metadata (shape, dtype, value range, mean per channel)
        """
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min": float(image.min()),
            "max": float(image.max()),
            "mean": [float(image[:, :, c].mean()) for c in range(image.shape[2])]
                    if len(image.shape) == 3 else float(image.mean()),
        }
