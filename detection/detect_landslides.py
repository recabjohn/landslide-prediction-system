"""
Landslide Detection Module
============================
Runs YOLOv8 inference on satellite images to detect:
  - Landslides
  - Debris flows
  - Normal terrain features

Supports both trained custom models and demo simulation mode.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Class names matching dataset.yaml
CLASS_NAMES = {0: "landslide", 1: "debris_flow", 2: "normal_terrain"}

# Color scheme per class (BGR)
CLASS_COLORS = {
    "landslide": (0, 0, 255),       # Red
    "debris_flow": (0, 140, 255),    # Orange
    "normal_terrain": (0, 200, 0),   # Green
}


class LandslideDetector:
    """
    YOLOv8-based landslide detection engine.

    In production mode, loads a trained YOLO model for inference.
    In demo mode, generates realistic simulated detections.
    """

    DEFAULT_WEIGHTS = [
        "models/landslide_detector_v1.pt",
        "models/landslide_detector_last.pt",
        "runs/landslide/india_terrain_v1/weights/best.pt",
    ]

    def __init__(self, model_path=None, demo_mode=False):
        """
        Initialize the detector.

        Args:
            model_path (str, optional): Path to trained YOLO weights (.pt file).
                If None, auto-discovers weights from models/ directory.
            demo_mode (bool): If True, use simulated detections instead of YOLO
        """
        self.model = None
        self.model_path = model_path
        self.demo_mode = demo_mode
        self.class_names = CLASS_NAMES

        # Fix PyTorch 2.6 weights_only compatibility
        self._patch_torch_load()

        if not demo_mode and model_path:
            self.load_model(model_path)
        elif not demo_mode:
            # Auto-discover trained weights
            found = self._auto_discover_weights()
            if not found:
                logger.info("No trained model found — switching to demo mode")
                self.demo_mode = True

    @staticmethod
    def _patch_torch_load():
        """Patch torch.load for PyTorch 2.6+ compatibility with YOLO weights."""
        try:
            import torch
            _orig = torch.load
            def _safe_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig(*args, **kwargs)
            torch.load = _safe_load
        except Exception:
            pass

    def _auto_discover_weights(self):
        """Try to find and load trained model weights automatically."""
        import pathlib
        project_root = pathlib.Path(__file__).parent.parent

        for rel_path in self.DEFAULT_WEIGHTS:
            full_path = project_root / rel_path
            if full_path.is_file():
                logger.info(f"Auto-discovered weights: {full_path}")
                return self.load_model(str(full_path))

        return False

    # ----------------------------------------------------------------
    # MODEL LOADING
    # ----------------------------------------------------------------

    def load_model(self, weights_path):
        """
        Load a trained YOLOv8 model.

        Args:
            weights_path (str): Path to the .pt weights file

        Returns:
            bool: True if model loaded successfully
        """
        try:
            from ultralytics import YOLO
            if not os.path.isfile(weights_path):
                logger.warning(f"Weights file not found: {weights_path}. Using demo mode.")
                self.demo_mode = True
                return False

            self.model = YOLO(weights_path)
            self.model_path = weights_path
            self.demo_mode = False
            logger.info(f"YOLO model loaded from: {weights_path}")
            return True

        except ImportError:
            logger.error("'ultralytics' package not installed. Using demo mode.")
            self.demo_mode = True
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Using demo mode.")
            self.demo_mode = True
            return False

    # ----------------------------------------------------------------
    # DETECTION
    # ----------------------------------------------------------------

    def detect(self, image, conf_threshold=0.25):
        """
        Detect landslides in a satellite image.

        Args:
            image (np.ndarray): Input image (BGR, uint8)
            conf_threshold (float): Minimum confidence threshold (0-1)

        Returns:
            list[dict]: List of detections, each containing:
                - 'class_id' (int): Class index
                - 'class_name' (str): Human-readable class name
                - 'confidence' (float): Detection confidence (0-1)
                - 'bbox' (list): [x1, y1, x2, y2] bounding box coordinates
                - 'area' (float): Bounding box area in pixels
        """
        if self.demo_mode:
            return self._simulate_detections(image, conf_threshold)

        return self._yolo_detect(image, conf_threshold)

    def _yolo_detect(self, image, conf_threshold):
        """Run actual YOLO inference."""
        results = self.model(image, conf=conf_threshold, verbose=False)
        detections = self.parse_results(results)
        logger.info(f"YOLO detected {len(detections)} objects (conf >= {conf_threshold})")
        return detections

    def parse_results(self, results):
        """
        Parse YOLO results into a standardized format.

        Args:
            results: Ultralytics YOLO results object

        Returns:
            list[dict]: Parsed detections
        """
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Map class ID to name
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                # Calculate area
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 1) for v in bbox],
                    "area": round(area, 1),
                })

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    # ----------------------------------------------------------------
    # DEMO / SIMULATION
    # ----------------------------------------------------------------

    def _simulate_detections(self, image, conf_threshold, seed=None):
        """
        Generate realistic simulated detections for demo mode.

        Analyzes the image to place detections in plausible locations
        (darker/browner regions that could be landslide scars).

        Args:
            image (np.ndarray): Input BGR image
            conf_threshold (float): Minimum confidence threshold
            seed (int, optional): Random seed

        Returns:
            list[dict]: Simulated detections
        """
        if seed is not None:
            np.random.seed(seed)

        h, w = image.shape[:2]
        detections = []

        # Analyze image to find candidate landslide regions
        # (regions with brownish/grey tones — reduced vegetation)
        candidates = self._find_candidate_regions(image)

        # Generate 2-4 detections
        num_detections = np.random.randint(2, 5)

        for i in range(min(num_detections, len(candidates) + 1)):
            # Use a candidate region if available, else random
            if i < len(candidates):
                cx, cy, region_size = candidates[i]
            else:
                cx = np.random.randint(w // 6, 5 * w // 6)
                cy = np.random.randint(h // 6, 5 * h // 6)
                region_size = np.random.randint(40, 120)

            # Generate bounding box
            box_w = int(region_size * np.random.uniform(0.8, 1.5))
            box_h = int(region_size * np.random.uniform(1.0, 2.0))
            x1 = max(0, cx - box_w // 2)
            y1 = max(0, cy - box_h // 2)
            x2 = min(w, x1 + box_w)
            y2 = min(h, y1 + box_h)

            # Assign class based on region characteristics
            if i == 0:
                cls_id = 0  # Primary detection is always "landslide"
                conf = np.random.uniform(0.82, 0.96)
            elif i == 1:
                cls_id = 1  # Secondary is "debris_flow"
                conf = np.random.uniform(0.70, 0.90)
            else:
                cls_id = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
                conf = np.random.uniform(0.45, 0.80)

            if conf >= conf_threshold:
                area = (x2 - x1) * (y2 - y1)
                detections.append({
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    "confidence": round(float(conf), 4),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "area": float(area),
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        logger.info(f"Demo mode: simulated {len(detections)} detections")
        return detections

    def _find_candidate_regions(self, image, grid_size=4):
        """
        Find image regions likely to contain landslide-like features.

        Looks for brownish/grey patches (bare earth) that contrast
        with surrounding vegetation (green patches).

        Args:
            image (np.ndarray): BGR image
            grid_size (int): Grid divisions for region analysis

        Returns:
            list[tuple]: (center_x, center_y, size) of candidate regions
        """
        import cv2
        h, w = image.shape[:2]
        candidates = []

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        cell_h = h // grid_size
        cell_w = w // grid_size

        for gy in range(grid_size):
            for gx in range(grid_size):
                y1 = gy * cell_h
                x1 = gx * cell_w
                y2 = min(y1 + cell_h, h)
                x2 = min(x1 + cell_w, w)

                cell = hsv[y1:y2, x1:x2]

                # Landslide indicators:
                #  - Low saturation (grey/brown)
                #  - Medium value (not too dark or bright)
                mean_sat = cell[:, :, 1].mean()
                mean_val = cell[:, :, 2].mean()

                # Low saturation = less vegetation = potential landslide
                if mean_sat < 80 and 60 < mean_val < 180:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    size = min(cell_w, cell_h)
                    candidates.append((cx, cy, size))

        # Sort by lowest saturation (most likely bare earth)
        candidates.sort(key=lambda c: c[2])
        return candidates[:4]

    # ----------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------

    def get_detection_summary(self, detections):
        """
        Generate a summary of detection results.

        Args:
            detections (list[dict]): List of detections

        Returns:
            dict: Summary statistics
        """
        if not detections:
            return {
                "total_detections": 0,
                "classes": {},
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "total_area": 0.0,
            }

        class_counts = {}
        for d in detections:
            cls = d["class_name"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        confidences = [d["confidence"] for d in detections]
        areas = [d["area"] for d in detections]

        return {
            "total_detections": len(detections),
            "classes": class_counts,
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
            "max_confidence": round(max(confidences), 4),
            "total_area": round(sum(areas), 1),
        }

    @staticmethod
    def get_class_color(class_name):
        """Get the display color (BGR) for a detection class."""
        return CLASS_COLORS.get(class_name, (200, 200, 200))
