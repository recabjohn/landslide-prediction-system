"""
Landslide Risk Estimation Module
==================================
Analyzes detection results and terrain characteristics to estimate
the probability of landslide risk in a given satellite image.

The risk score is computed from multiple factors:
  1. Detection-based features (count, confidence, area coverage)
  2. Terrain analysis (color distribution, texture, elevation patterns)
  3. Spatial distribution of detected hazards

Output: Risk score (0-100%) and risk level (Low/Medium/High/Critical)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Risk level thresholds
RISK_LEVELS = {
    "Low":      (0, 25),
    "Medium":   (25, 50),
    "High":     (50, 75),
    "Critical": (75, 100),
}

# Risk level colors (BGR)
RISK_COLORS = {
    "Low":      (0, 180, 0),      # Green
    "Medium":   (0, 200, 255),    # Yellow
    "High":     (0, 100, 255),    # Orange
    "Critical": (0, 0, 255),      # Red
}

# Risk level emoji
RISK_EMOJI = {
    "Low":      "🟢",
    "Medium":   "🟡",
    "High":     "🟠",
    "Critical": "🔴",
}


class RiskEstimator:
    """
    Estimates landslide risk from detection results and image features.

    The risk estimation combines:
      - Detection score: based on what the YOLO model found
      - Terrain score: based on image-level terrain analysis
      - Spatial score: based on distribution pattern of detections
    """

    # Weighting of risk components
    WEIGHT_DETECTION = 0.50  # 50% from detection results
    WEIGHT_TERRAIN = 0.30    # 30% from terrain analysis
    WEIGHT_SPATIAL = 0.20    # 20% from spatial distribution

    def __init__(self):
        """Initialize the risk estimator."""
        self._last_breakdown = None

    # ----------------------------------------------------------------
    # MAIN API
    # ----------------------------------------------------------------

    def estimate_risk(self, detections, image=None, image_shape=None):
        """
        Compute the overall landslide risk score.

        Args:
            detections (list[dict]): Detection results from LandslideDetector.
                Each dict must contain: 'class_name', 'confidence', 'bbox', 'area'
            image (np.ndarray, optional): Original BGR image for terrain analysis.
                If not provided, terrain score defaults to 0.
            image_shape (tuple, optional): (H, W) of the image. Used if image
                is not provided for area normalization.

        Returns:
            tuple: (risk_score: float 0-100, risk_level: str)
        """
        # Determine image dimensions
        if image is not None:
            h, w = image.shape[:2]
        elif image_shape is not None:
            h, w = image_shape[:2]
        else:
            h, w = 640, 640  # default

        total_image_area = h * w

        # --- Component 1: Detection-based risk ---
        detection_score = self._compute_detection_score(detections, total_image_area)

        # --- Component 2: Terrain-based risk ---
        terrain_score = self._compute_terrain_score(image) if image is not None else 0.0

        # --- Component 3: Spatial distribution risk ---
        spatial_score = self._compute_spatial_score(detections, h, w)

        # --- Weighted combination ---
        raw_score = (
            self.WEIGHT_DETECTION * detection_score +
            self.WEIGHT_TERRAIN * terrain_score +
            self.WEIGHT_SPATIAL * spatial_score
        )

        # Clamp to [0, 100]
        risk_score = round(min(max(raw_score, 0), 100), 1)
        risk_level = self._score_to_level(risk_score)

        # Store breakdown for reporting
        self._last_breakdown = {
            "detection_score": round(detection_score, 1),
            "terrain_score": round(terrain_score, 1),
            "spatial_score": round(spatial_score, 1),
            "weights": {
                "detection": self.WEIGHT_DETECTION,
                "terrain": self.WEIGHT_TERRAIN,
                "spatial": self.WEIGHT_SPATIAL,
            },
            "risk_score": risk_score,
            "risk_level": risk_level,
        }

        logger.info(
            f"Risk estimation: score={risk_score}%, level={risk_level} "
            f"(det={detection_score:.1f}, terrain={terrain_score:.1f}, spatial={spatial_score:.1f})"
        )

        return risk_score, risk_level

    # ----------------------------------------------------------------
    # DETECTION SCORE
    # ----------------------------------------------------------------

    def _compute_detection_score(self, detections, total_image_area):
        """
        Score based on detection results.

        Factors:
          - Number of hazard detections (landslide + debris_flow)
          - Average confidence of hazard detections
          - Area coverage ratio (detected area / total image area)

        Args:
            detections (list[dict]): Detection results
            total_image_area (float): Total image area in pixels

        Returns:
            float: Detection risk score (0-100)
        """
        if not detections:
            return 0.0

        # Filter for hazard classes only
        hazard_classes = {"landslide", "debris_flow"}
        hazards = [d for d in detections if d["class_name"] in hazard_classes]

        if not hazards:
            return 5.0  # Minimal base risk if no hazards detected

        # Factor 1: Number of detections (more = higher risk)
        count_score = min(len(hazards) * 20, 40)  # Max 40 from count

        # Factor 2: Average confidence
        avg_conf = sum(d["confidence"] for d in hazards) / len(hazards)
        conf_score = avg_conf * 35  # Max 35 from confidence

        # Factor 3: Area coverage
        total_hazard_area = sum(d.get("area", 0) for d in hazards)
        coverage_ratio = total_hazard_area / total_image_area if total_image_area > 0 else 0
        area_score = min(coverage_ratio * 500, 25)  # Max 25 from area

        # Higher score for landslides vs debris flows
        landslide_bonus = 0
        for d in hazards:
            if d["class_name"] == "landslide" and d["confidence"] > 0.8:
                landslide_bonus += 5

        return min(count_score + conf_score + area_score + landslide_bonus, 100)

    # ----------------------------------------------------------------
    # TERRAIN SCORE
    # ----------------------------------------------------------------

    def _compute_terrain_score(self, image):
        """
        Analyze terrain features from the image.

        Factors:
          - Vegetation coverage (green regions)
          - Bare earth / exposed soil (brown/grey regions)
          - Texture roughness (edge density)
          - Color uniformity deviation

        Args:
            image (np.ndarray): BGR image

        Returns:
            float: Terrain risk score (0-100)
        """
        import cv2

        h, w = image.shape[:2]
        total_pixels = h * w

        # Convert to HSV for vegetation analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- Factor 1: Vegetation coverage (lower = higher risk) ---
        # Green vegetation: H in [30-80], S > 40, V > 40
        green_mask = cv2.inRange(hsv, (30, 40, 40), (80, 255, 255))
        vegetation_ratio = np.sum(green_mask > 0) / total_pixels
        # Less vegetation → higher risk
        veg_risk = (1 - vegetation_ratio) * 35  # Max 35

        # --- Factor 2: Bare earth detection ---
        # Brown/grey: H in [10-25] OR low saturation
        brown_mask = cv2.inRange(hsv, (10, 30, 50), (25, 200, 200))
        grey_mask = cv2.inRange(hsv, (0, 0, 60), (180, 40, 200))
        bare_earth = cv2.bitwise_or(brown_mask, grey_mask)
        bare_ratio = np.sum(bare_earth > 0) / total_pixels
        bare_risk = bare_ratio * 30  # Max 30

        # --- Factor 3: Texture roughness (high edges = rough terrain) ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        texture_risk = min(edge_density * 200, 20)  # Max 20

        # --- Factor 4: Color variance (high variance = disturbed terrain) ---
        channel_stds = [image[:, :, c].std() for c in range(3)]
        avg_std = sum(channel_stds) / 3
        variance_risk = min(avg_std / 60 * 15, 15)  # Max 15

        terrain_score = veg_risk + bare_risk + texture_risk + variance_risk
        return min(terrain_score, 100)

    # ----------------------------------------------------------------
    # SPATIAL SCORE
    # ----------------------------------------------------------------

    def _compute_spatial_score(self, detections, img_h, img_w):
        """
        Analyze the spatial distribution of detections.

        Clustered detections suggest a higher risk than dispersed ones.

        Args:
            detections (list[dict]): Detection results
            img_h (int): Image height
            img_w (int): Image width

        Returns:
            float: Spatial risk score (0-100)
        """
        hazard_classes = {"landslide", "debris_flow"}
        hazards = [d for d in detections if d["class_name"] in hazard_classes]

        if len(hazards) < 2:
            return len(hazards) * 20  # Simple score for 0-1 detections

        # Get centers of all hazard detections
        centers = []
        for d in hazards:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))

        # Calculate average pairwise distance (normalized by image diagonal)
        diagonal = np.sqrt(img_h ** 2 + img_w ** 2)
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt(
                    (centers[i][0] - centers[j][0]) ** 2 +
                    (centers[i][1] - centers[j][1]) ** 2
                )
                distances.append(dist / diagonal)

        avg_distance = sum(distances) / len(distances) if distances else 0

        # Closer detections (clustered) = higher risk
        clustering_score = (1 - avg_distance) * 60  # Max 60

        # More detections = higher spatial risk
        count_bonus = min(len(hazards) * 10, 40)  # Max 40

        return min(clustering_score + count_bonus, 100)

    # ----------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------

    @staticmethod
    def _score_to_level(score):
        """Convert a numeric risk score to a risk level string."""
        for level, (low, high) in RISK_LEVELS.items():
            if low <= score < high:
                return level
        return "Critical"  # score >= 75

    def get_risk_breakdown(self):
        """
        Get the detailed breakdown of the last risk estimation.

        Returns:
            dict: Breakdown of risk components, or None if no estimation done
        """
        return self._last_breakdown

    @staticmethod
    def get_risk_color(risk_level):
        """Get the display color (BGR) for a risk level."""
        return RISK_COLORS.get(risk_level, (200, 200, 200))

    @staticmethod
    def get_risk_emoji(risk_level):
        """Get the emoji for a risk level."""
        return RISK_EMOJI.get(risk_level, "⚪")

    @staticmethod
    def format_risk_report(risk_score, risk_level, breakdown=None):
        """
        Format a human-readable risk report.

        Args:
            risk_score (float): Overall risk score (0-100)
            risk_level (str): Risk level string
            breakdown (dict, optional): Detailed breakdown

        Returns:
            str: Formatted report string
        """
        emoji = RISK_EMOJI.get(risk_level, "⚪")
        report = [
            f"{'='*50}",
            f"  LANDSLIDE RISK ASSESSMENT REPORT",
            f"{'='*50}",
            f"",
            f"  {emoji} Risk Level : {risk_level}",
            f"  📊 Risk Score : {risk_score}%",
            f"",
        ]

        if breakdown:
            report.extend([
                f"  --- Score Breakdown ---",
                f"  Detection Score : {breakdown.get('detection_score', 'N/A')}",
                f"  Terrain Score   : {breakdown.get('terrain_score', 'N/A')}",
                f"  Spatial Score   : {breakdown.get('spatial_score', 'N/A')}",
                f"",
            ])

        report.append(f"{'='*50}")
        return "\n".join(report)
