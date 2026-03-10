"""
Demo Sample Generator
=======================
Generates procedural satellite-like images with terrain features
for demonstrating the landslide detection system without real data.
"""

import os
import json
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def generate_terrain_image(size=640, seed=None, style="mountainous"):
    """
    Generate a procedural satellite-like terrain image.

    Args:
        size (int): Image dimensions (square)
        seed (int): Random seed for reproducibility
        style (str): Terrain style ('mountainous', 'hilly', 'valley')

    Returns:
        np.ndarray: BGR image (size, size, 3)
    """
    if seed is not None:
        np.random.seed(seed)

    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Style-dependent base colors
    styles = {
        "mountainous": {"base_g": (60, 100), "base_r": (70, 120), "base_b": (35, 55)},
        "hilly":       {"base_g": (80, 130), "base_r": (55, 95),  "base_b": (40, 60)},
        "valley":      {"base_g": (90, 140), "base_r": (50, 80),  "base_b": (45, 65)},
    }
    s = styles.get(style, styles["mountainous"])

    # Base terrain with gradient
    for y in range(size):
        ratio = y / size
        g = int(s["base_g"][0] + (s["base_g"][1] - s["base_g"][0]) * (1 - ratio))
        r = int(s["base_r"][0] + (s["base_r"][1] - s["base_r"][0]) * ratio)
        b = int(s["base_b"][0] + (s["base_b"][1] - s["base_b"][0]) * (1 - ratio))
        img[y, :] = [b, g, r]

    # Natural noise
    noise = np.random.randint(-12, 12, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Mountain ridges
    num_ridges = np.random.randint(2, 5)
    for _ in range(num_ridges):
        ridge_y = np.random.randint(size // 5, 4 * size // 5)
        freq = np.random.uniform(20, 60)
        amplitude = np.random.randint(15, 40)
        for x in range(size):
            y_offset = int(amplitude * np.sin(x / freq + np.random.random() * 0.3))
            y_pos = ridge_y + y_offset
            if 2 <= y_pos < size - 4:
                img[y_pos:y_pos + 2, x] = [35, 45, 40]
                img[y_pos - 2:y_pos, x] = [70, 110, 90]

    # Vegetation patches
    for _ in range(np.random.randint(15, 35)):
        vx = np.random.randint(0, size - 25)
        vy = np.random.randint(0, size - 25)
        vs = np.random.randint(5, 25)
        green_add = np.random.randint(20, 60)
        region = img[vy:vy + vs, vx:vx + vs].copy()
        region[:, :, 1] = np.clip(region[:, :, 1].astype(np.int16) + green_add, 0, 255).astype(np.uint8)
        img[vy:vy + vs, vx:vx + vs] = region

    # Water features (thin blue lines)
    if np.random.random() > 0.4:
        water_x = np.random.randint(size // 3, 2 * size // 3)
        for y in range(size):
            wx = water_x + int(15 * np.sin(y / 30))
            if 0 <= wx < size - 2:
                img[y, wx:wx + 2] = [180, 140, 80]

    return img


def add_landslide_scar(img, x=None, y=None, w=None, h=None):
    """
    Add a landslide scar feature to the image.

    Returns:
        tuple: (modified image, (x, y, w, h) of scar)
    """
    size = img.shape[0]
    x = x or np.random.randint(size // 6, 2 * size // 3)
    y = y or np.random.randint(size // 6, 2 * size // 3)
    w = w or np.random.randint(70, 150)
    h = h or np.random.randint(90, 200)

    result = img.copy()

    for py in range(max(0, y), min(size, y + h)):
        for px in range(max(0, x), min(size, x + w)):
            # Elliptical scar shape
            dist = ((px - x - w // 2) ** 2 / max((w // 2) ** 2, 1) +
                    (py - y - h // 2) ** 2 / max((h // 2) ** 2, 1))
            if dist < 1.0 + 0.2 * np.random.random():
                result[py, px] = [
                    np.clip(int(85 + np.random.randint(-15, 15)), 0, 255),
                    np.clip(int(95 + np.random.randint(-15, 15)), 0, 255),
                    np.clip(int(115 + np.random.randint(-15, 15)), 0, 255),
                ]

    return result, (x, y, w, h)


def add_debris_flow(img, x=None, y=None, length=None):
    """
    Add a debris flow channel feature.

    Returns:
        tuple: (modified image, (x, y, end_x, end_y) bounding box)
    """
    size = img.shape[0]
    x = x or np.random.randint(size // 4, 3 * size // 4)
    y = y or np.random.randint(size // 4, size // 2)
    length = length or np.random.randint(60, 120)

    result = img.copy()
    min_x, min_y = x, y
    max_x, max_y = x, y

    cx, cy = x, y
    for i in range(length):
        cx += int(np.random.uniform(-2, 3))
        cy += int(np.random.uniform(1, 3))

        if 0 <= cx < size - 5 and 0 <= cy < size - 5:
            spread = max(2, 6 - i // 25)
            result[cy:cy + spread, cx:cx + spread] = [
                np.clip(int(65 + np.random.randint(-10, 10)), 0, 255),
                np.clip(int(75 + np.random.randint(-10, 10)), 0, 255),
                np.clip(int(90 + np.random.randint(-10, 10)), 0, 255),
            ]
            min_x = min(min_x, cx)
            min_y = min(min_y, cy)
            max_x = max(max_x, cx + spread)
            max_y = max(max_y, cy + spread)

    return result, (min_x, min_y, max_x - min_x, max_y - min_y)


def generate_demo_samples(output_dir, num_samples=5):
    """
    Generate a set of demo satellite images with pre-computed detection results.

    Args:
        output_dir (str): Directory to save images and results
        num_samples (int): Number of samples to generate

    Returns:
        list[dict]: Sample metadata with file paths and detection results
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    styles = ["mountainous", "hilly", "valley", "mountainous", "hilly"]
    samples = []

    for i in range(num_samples):
        seed = 42 + i
        style = styles[i % len(styles)]

        # Generate base terrain
        img = generate_terrain_image(size=640, seed=seed, style=style)

        detections = []

        # Add landslide scar
        img, (lx, ly, lw, lh) = add_landslide_scar(img)
        detections.append({
            "class_id": 0,
            "class_name": "landslide",
            "confidence": round(float(np.random.uniform(0.82, 0.96)), 4),
            "bbox": [float(lx), float(ly), float(lx + lw), float(ly + lh)],
            "area": float(lw * lh),
        })

        # Maybe add debris flow
        if np.random.random() > 0.3:
            img, (dx, dy, dw, dh) = add_debris_flow(img)
            detections.append({
                "class_id": 1,
                "class_name": "debris_flow",
                "confidence": round(float(np.random.uniform(0.68, 0.88)), 4),
                "bbox": [float(dx), float(dy), float(dx + dw), float(dy + dh)],
                "area": float(dw * dh),
            })

        # Maybe add a second smaller landslide
        if np.random.random() > 0.5:
            img, (lx2, ly2, lw2, lh2) = add_landslide_scar(
                img,
                x=np.random.randint(320, 520),
                y=np.random.randint(320, 480),
                w=np.random.randint(40, 80),
                h=np.random.randint(50, 90),
            )
            detections.append({
                "class_id": 0,
                "class_name": "landslide",
                "confidence": round(float(np.random.uniform(0.55, 0.78)), 4),
                "bbox": [float(lx2), float(ly2), float(lx2 + lw2), float(ly2 + lh2)],
                "area": float(lw2 * lh2),
            })

        # Save image
        filename = f"sample_{i + 1:02d}_{style}.png"
        filepath = os.path.join(images_dir, filename)
        img_rgb = img[:, :, ::-1]  # BGR → RGB
        Image.fromarray(img_rgb).save(filepath)

        samples.append({
            "filename": filename,
            "filepath": filepath,
            "style": style,
            "seed": seed,
            "detections": detections,
        })

        logger.info(f"Generated sample {i + 1}/{num_samples}: {filename}")

    # Save metadata
    results_path = os.path.join(output_dir, "sample_results.json")
    with open(results_path, "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Saved {num_samples} demo samples to {output_dir}")
    return samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demo_dir = os.path.join(project_root, "demo")
    generate_demo_samples(demo_dir, num_samples=5)
    print(f"\n✅ Demo samples generated in: {demo_dir}")
