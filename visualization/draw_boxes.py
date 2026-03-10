"""
Visualization Module
=====================
Renders detection results on satellite images with:
  - Color-coded bounding boxes per class
  - Confidence score labels
  - Risk level overlay badges
  - Summary charts and statistics plots
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Detection class colors (BGR)
CLASS_COLORS = {
    "landslide":       (0, 0, 230),      # Red
    "debris_flow":     (0, 140, 255),     # Orange
    "normal_terrain":  (0, 200, 0),       # Green
}

# Risk level colors (BGR)
RISK_COLORS = {
    "Low":      (0, 180, 0),
    "Medium":   (0, 200, 255),
    "High":     (0, 100, 255),
    "Critical": (0, 0, 255),
}

RISK_BADGE_TEXT = {
    "Low":      "LOW RISK",
    "Medium":   "MEDIUM RISK",
    "High":     "HIGH RISK",
    "Critical": "CRITICAL RISK",
}


class Visualizer:
    """
    Draws detection results and risk information on satellite images.
    """

    def __init__(self, font_scale=0.6, thickness=2, alpha=0.3):
        """
        Initialize the visualizer.

        Args:
            font_scale (float): Text font scale
            thickness (int): Line/box thickness
            alpha (float): Overlay transparency (0-1)
        """
        self.font_scale = font_scale
        self.thickness = thickness
        self.alpha = alpha
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # ----------------------------------------------------------------
    # DETECTION VISUALIZATION
    # ----------------------------------------------------------------

    def draw_detections(self, image, detections, show_labels=True, show_confidence=True):
        """
        Draw bounding boxes and labels for all detections.

        Args:
            image (np.ndarray): Input BGR image (will not be modified)
            detections (list[dict]): Detection results with 'bbox', 'class_name', 'confidence'
            show_labels (bool): Show class name labels
            show_confidence (bool): Show confidence scores

        Returns:
            np.ndarray: Annotated image copy
        """
        annotated = image.copy()

        for det in detections:
            bbox = det["bbox"]
            cls_name = det.get("class_name", "unknown")
            confidence = det.get("confidence", 0.0)
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))

            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.thickness)

            # Draw semi-transparent fill
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            annotated = cv2.addWeighted(overlay, self.alpha * 0.3, annotated, 1 - self.alpha * 0.3, 0)

            # Build label text
            if show_labels or show_confidence:
                parts = []
                if show_labels:
                    parts.append(cls_name.replace("_", " ").title())
                if show_confidence:
                    parts.append(f"{confidence:.0%}")
                label = " | ".join(parts)

                # Draw label background
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, self.font, self.font_scale, 1
                )
                label_y = max(y1 - 8, label_h + 4)

                cv2.rectangle(
                    annotated,
                    (x1, label_y - label_h - 6),
                    (x1 + label_w + 8, label_y + 2),
                    color, -1
                )

                # Draw label text (white on colored background)
                cv2.putText(
                    annotated, label,
                    (x1 + 4, label_y - 4),
                    self.font, self.font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA
                )

        logger.info(f"Drew {len(detections)} detection boxes")
        return annotated

    # ----------------------------------------------------------------
    # RISK OVERLAY
    # ----------------------------------------------------------------

    def overlay_risk(self, image, risk_score, risk_level, position="top-right"):
        """
        Overlay a risk badge on the image.

        Args:
            image (np.ndarray): Input BGR image (will not be modified)
            risk_score (float): Risk score (0-100)
            risk_level (str): Risk level ('Low', 'Medium', 'High', 'Critical')
            position (str): Badge position ('top-right', 'top-left', 'bottom-right', 'bottom-left')

        Returns:
            np.ndarray: Image with risk badge overlay
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        color = RISK_COLORS.get(risk_level, (200, 200, 200))
        badge_text = RISK_BADGE_TEXT.get(risk_level, risk_level.upper())
        score_text = f"{risk_score:.0f}%"

        # Badge dimensions
        badge_w, badge_h = 200, 70
        padding = 10

        # Position
        positions = {
            "top-right":    (w - badge_w - padding, padding),
            "top-left":     (padding, padding),
            "bottom-right": (w - badge_w - padding, h - badge_h - padding),
            "bottom-left":  (padding, h - badge_h - padding),
        }
        bx, by = positions.get(position, positions["top-right"])

        # Draw badge background (semi-transparent)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), color, -1)
        annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)

        # Badge border
        cv2.rectangle(annotated, (bx, by), (bx + badge_w, by + badge_h), color, 2)

        # Badge text
        cv2.putText(
            annotated, badge_text,
            (bx + 10, by + 25),
            self.font, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            annotated, f"Score: {score_text}",
            (bx + 10, by + 52),
            self.font, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )

        logger.info(f"Risk badge overlaid: {risk_level} ({risk_score}%)")
        return annotated

    # ----------------------------------------------------------------
    # COMBINED VIEW
    # ----------------------------------------------------------------

    def create_annotated_image(self, image, detections, risk_score, risk_level):
        """
        Create a fully annotated satellite image with detections and risk overlay.

        Args:
            image (np.ndarray): Original BGR satellite image
            detections (list[dict]): Detection results
            risk_score (float): Risk score (0-100)
            risk_level (str): Risk level string

        Returns:
            np.ndarray: Fully annotated image
        """
        # Step 1: Draw detection boxes
        result = self.draw_detections(image, detections)

        # Step 2: Add risk badge
        result = self.overlay_risk(result, risk_score, risk_level)

        return result

    # ----------------------------------------------------------------
    # MATPLOTLIB CHARTS
    # ----------------------------------------------------------------

    def create_summary_plot(self, detections, risk_score, risk_level):
        """
        Create a matplotlib summary figure with detection statistics.

        Args:
            detections (list[dict]): Detection results
            risk_score (float): Risk score (0-100)
            risk_level (str): Risk level string

        Returns:
            matplotlib.figure.Figure: Summary plot figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("#1a1a2e")

        # --- Chart 1: Detection class distribution ---
        ax1 = axes[0]
        class_counts = {}
        for d in detections:
            cls = d["class_name"].replace("_", " ").title()
            class_counts[cls] = class_counts.get(cls, 0) + 1

        if class_counts:
            colors_map = {
                "Landslide": "#ff4444",
                "Debris Flow": "#ff9800",
                "Normal Terrain": "#4caf50",
            }
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            colors = [colors_map.get(c, "#888888") for c in classes]

            bars = ax1.barh(classes, counts, color=colors, edgecolor="white", linewidth=0.5)
            ax1.set_xlabel("Count", color="white", fontsize=10)
            ax1.set_title("Detection Classes", color="white", fontweight="bold", fontsize=12)
        else:
            ax1.text(0.5, 0.5, "No Detections", ha="center", va="center",
                     color="white", fontsize=14, transform=ax1.transAxes)
            ax1.set_title("Detection Classes", color="white", fontweight="bold", fontsize=12)

        ax1.set_facecolor("#16213e")
        ax1.tick_params(colors="white")
        for spine in ax1.spines.values():
            spine.set_color("#333")

        # --- Chart 2: Confidence scores ---
        ax2 = axes[1]
        if detections:
            confidences = [d["confidence"] for d in detections]
            labels = [d["class_name"].replace("_", " ").title()[:10] for d in detections]
            bar_colors = ["#ff4444" if c > 0.8 else "#ff9800" if c > 0.5 else "#4caf50"
                          for c in confidences]
            ax2.bar(range(len(confidences)), confidences, color=bar_colors,
                    edgecolor="white", linewidth=0.5)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Confidence", color="white", fontsize=10)
            ax2.axhline(y=0.8, color="#ff4444", linestyle="--", alpha=0.5, label="High conf.")
            ax2.axhline(y=0.5, color="#ff9800", linestyle="--", alpha=0.5, label="Med conf.")
        else:
            ax2.text(0.5, 0.5, "No Detections", ha="center", va="center",
                     color="white", fontsize=14, transform=ax2.transAxes)

        ax2.set_title("Confidence Scores", color="white", fontweight="bold", fontsize=12)
        ax2.set_facecolor("#16213e")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_color("#333")

        # --- Chart 3: Risk gauge ---
        ax3 = axes[2]
        gauge_colors = ["#4caf50", "#ffeb3b", "#ff9800", "#ff4444"]
        gauge_labels = ["Low", "Medium", "High", "Critical"]
        gauge_sizes = [25, 25, 25, 25]

        wedges, _ = ax3.pie(
            gauge_sizes, colors=gauge_colors,
            startangle=180, counterclock=False,
            wedgeprops={"width": 0.3, "edgecolor": "#1a1a2e", "linewidth": 2}
        )

        # Draw needle
        angle = 180 - (risk_score / 100) * 180
        needle_x = 0.35 * np.cos(np.radians(angle))
        needle_y = 0.35 * np.sin(np.radians(angle))
        ax3.annotate("", xy=(needle_x, needle_y), xytext=(0, 0),
                      arrowprops={"arrowstyle": "->", "color": "white", "lw": 2.5})

        # Center text
        ax3.text(0, -0.15, f"{risk_score:.0f}%", ha="center", va="center",
                 color="white", fontsize=20, fontweight="bold")
        risk_hex = {"Low": "#4caf50", "Medium": "#ffd200", "High": "#ff4444", "Critical": "#ea384d"}
        ax3.text(0, -0.32, risk_level, ha="center", va="center",
                 color=risk_hex.get(risk_level, "white"),
                 fontsize=12, fontweight="bold")

        ax3.set_title("Risk Score", color="white", fontweight="bold", fontsize=12)
        ax3.set_facecolor("#1a1a2e")

        plt.tight_layout()
        return fig

    # ----------------------------------------------------------------
    # SIDE-BY-SIDE COMPARISON
    # ----------------------------------------------------------------

    def create_comparison(self, original, annotated, title_left="Original", title_right="Detection Results"):
        """
        Create a side-by-side comparison image.

        Args:
            original (np.ndarray): Original satellite image
            annotated (np.ndarray): Annotated image with detections
            title_left (str): Title for the left image
            title_right (str): Title for the right image

        Returns:
            np.ndarray: Side-by-side comparison image
        """
        # Ensure same dimensions
        h1, w1 = original.shape[:2]
        h2, w2 = annotated.shape[:2]
        h = max(h1, h2)
        w = max(w1, w2)

        # Resize to match
        img1 = cv2.resize(original, (w, h))
        img2 = cv2.resize(annotated, (w, h))

        # Add title bars
        bar_h = 40
        bar1 = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar2 = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar1[:] = (50, 50, 50)
        bar2[:] = (50, 50, 50)

        cv2.putText(bar1, title_left, (10, 28), self.font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar2, title_right, (10, 28), self.font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        left = np.vstack([bar1, img1])
        right = np.vstack([bar2, img2])

        # Separator
        separator = np.full((h + bar_h, 3, 3), (0, 200, 255), dtype=np.uint8)

        comparison = np.hstack([left, separator, right])
        return comparison

    # ----------------------------------------------------------------
    # UTILITY
    # ----------------------------------------------------------------

    @staticmethod
    def save_image(image, path):
        """Save a BGR image to disk."""
        cv2.imwrite(path, image)
        logger.info(f"Image saved to {path}")

    @staticmethod
    def bgr_to_rgb(image):
        """Convert BGR to RGB for display in matplotlib/streamlit."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
