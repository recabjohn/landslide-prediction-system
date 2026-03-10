#!/usr/bin/env python3
"""
Dataset Generator for Landslide Detection
============================================
Generates a complete YOLO-format dataset for training the landslide
detection model. Uses multiple sources:

1. NASA GIBS API — real satellite imagery tiles (free, no key required)
2. NASA EONET API — real landslide event locations
3. High-quality procedural generation — synthetic satellite imagery

The generator creates:
  - Realistic satellite terrain images
  - Landslide scar, debris flow, and normal terrain features
  - YOLO-format annotation .txt files
  - Proper train/val/test directory structure

Usage:
    python generate_dataset.py --train 120 --val 30 --test 20
"""

import os
import sys
import json
import random
import logging
import argparse
import math
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

CLASSES = {0: "landslide", 1: "debris_flow", 2: "normal_terrain"}
IMG_SIZE = 640

# NASA GIBS WMTS — free satellite imagery tiles (no API key needed)
NASA_GIBS_URL = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
    "{layer}/default/{date}/{tilematrixset}/{z}/{y}/{x}.jpg"
)

# NASA EONET — landslide & natural disaster events (free, no key)
NASA_EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"

# Sentinel-2 Cloudless tiles (free)
SENTINEL_TILES_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg"


# ================================================================
# NASA API INTEGRATION
# ================================================================

class NASAImageFetcher:
    """
    Fetches real satellite imagery from NASA's free APIs.
    """

    # Indian landslide-prone regions — Himalayas, Western Ghats, Northeast
    LANDSLIDE_REGIONS = [
        # --- Western Ghats (Tamil Nadu / Kerala / Karnataka) ---
        {"name": "Kodaikanal Hills",          "lat": 10.2381, "lon": 77.4892},
        {"name": "Ooty Nilgiris",             "lat": 11.4102, "lon": 76.6950},
        {"name": "Coonoor Nilgiris",           "lat": 11.3530, "lon": 76.7959},
        {"name": "Munnar Kerala",              "lat": 10.0889, "lon": 77.0595},
        {"name": "Wayanad Kerala",             "lat": 11.6854, "lon": 76.1320},
        {"name": "Idukki Kerala",              "lat": 9.8494,  "lon": 76.9710},
        {"name": "Coorg Karnataka",            "lat": 12.3375, "lon": 75.8069},
        {"name": "Chikmagalur Karnataka",      "lat": 13.3161, "lon": 75.7720},
        {"name": "Valparai Tamil Nadu",        "lat": 10.3268, "lon": 76.9506},
        {"name": "Meghamalai Tamil Nadu",      "lat": 9.7200,  "lon": 77.4300},
        # --- Himalayas (Uttarakhand / HP / J&K / Sikkim) ---
        {"name": "Kedarnath Uttarakhand",      "lat": 30.7352, "lon": 79.0669},
        {"name": "Joshimath Uttarakhand",      "lat": 30.5550, "lon": 79.5650},
        {"name": "Chamoli Uttarakhand",        "lat": 30.4050, "lon": 79.3250},
        {"name": "Rishikesh Uttarakhand",      "lat": 30.0869, "lon": 78.2676},
        {"name": "Shimla Himachal",            "lat": 31.1048, "lon": 77.1734},
        {"name": "Manali Himachal",            "lat": 32.2396, "lon": 77.1887},
        {"name": "Dharamshala Himachal",       "lat": 32.2190, "lon": 76.3234},
        {"name": "Kullu Himachal",             "lat": 31.9579, "lon": 77.1095},
        {"name": "Gangtok Sikkim",             "lat": 27.3389, "lon": 88.6065},
        {"name": "Lachung Sikkim",             "lat": 27.6950, "lon": 88.7467},
        # --- Northeast India ---
        {"name": "Shillong Meghalaya",         "lat": 25.5788, "lon": 91.8933},
        {"name": "Cherrapunji Meghalaya",      "lat": 25.2700, "lon": 91.7200},
        {"name": "Aizawl Mizoram",             "lat": 23.7271, "lon": 92.7176},
        {"name": "Kohima Nagaland",            "lat": 25.6751, "lon": 94.1086},
        {"name": "Itanagar Arunachal",         "lat": 27.0844, "lon": 93.6053},
        # --- Other Indian regions ---
        {"name": "Darjeeling West Bengal",     "lat": 27.0360, "lon": 88.2627},
        {"name": "Pune Western Ghats",         "lat": 18.7167, "lon": 73.6833},
        {"name": "Mahabaleshwar Maharashtra",  "lat": 17.9307, "lon": 73.6477},
        {"name": "Kalsubai Maharashtra",       "lat": 19.6000, "lon": 73.7100},
        {"name": "Jammu Kashmir Pir Panjal",   "lat": 33.8880, "lon": 74.7973},
    ]

    @staticmethod
    def fetch_eonet_landslides():
        """
        Fetch recent landslide events from NASA EONET API.

        Returns:
            list[dict]: Landslide events with coordinates
        """
        if not HAS_REQUESTS:
            return []
        try:
            params = {"category": "landslides", "status": "open", "limit": 50}
            resp = requests.get(NASA_EONET_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            events = []
            for event in data.get("events", []):
                for geom in event.get("geometry", []):
                    coords = geom.get("coordinates", [])
                    if len(coords) >= 2:
                        events.append({
                            "name": event.get("title", "Unknown"),
                            "lat": coords[1],
                            "lon": coords[0],
                            "date": geom.get("date", ""),
                        })
            logger.info(f"NASA EONET: Found {len(events)} landslide events")
            return events
        except Exception as e:
            logger.warning(f"NASA EONET fetch failed: {e}")
            return []

    @staticmethod
    def fetch_satellite_tile(lat, lon, zoom=12, source="sentinel"):
        """
        Fetch a satellite image tile for given coordinates.

        Args:
            lat (float): Latitude
            lon (float): Longitude
            zoom (int): Zoom level (higher = more detail)
            source (str): 'sentinel' or 'nasa_gibs'

        Returns:
            PIL.Image or None
        """
        if not HAS_REQUESTS:
            return None

        # Convert lat/lon to tile coordinates
        n = 2 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.log(math.tan(math.radians(lat)) +
                    1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)

        try:
            if source == "sentinel":
                url = SENTINEL_TILES_URL.format(z=zoom, y=y_tile, x=x_tile)
            else:
                url = NASA_GIBS_URL.format(
                    layer="MODIS_Terra_CorrectedReflectance_TrueColor",
                    date="2024-06-15",
                    tilematrixset="250m",
                    z=zoom, y=y_tile, x=x_tile,
                )

            resp = requests.get(url, timeout=20, headers={
                "User-Agent": "LandslideDetection/1.0 (Academic Project)"
            })
            resp.raise_for_status()

            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            logger.info(f"Fetched satellite tile at ({lat:.2f}, {lon:.2f}) from {source}")
            return img

        except Exception as e:
            logger.debug(f"Tile fetch failed ({source}): {e}")
            return None

    @classmethod
    def fetch_batch(cls, count=20, zoom=12):
        """
        Fetch a batch of satellite images from various landslide regions.

        Args:
            count (int): Number of images to fetch
            zoom (int): Zoom level

        Returns:
            list[PIL.Image]: Fetched satellite images
        """
        images = []
        regions = cls.LANDSLIDE_REGIONS.copy()
        random.shuffle(regions)

        # Also try NASA EONET for real event locations
        eonet_events = cls.fetch_eonet_landslides()

        all_locations = eonet_events + regions

        for loc in all_locations[:count * 3]:  # Try more than needed
            if len(images) >= count:
                break

            lat = loc["lat"] + random.uniform(-0.05, 0.05)
            lon = loc["lon"] + random.uniform(-0.05, 0.05)

            # Try Sentinel first, then NASA GIBS
            img = cls.fetch_satellite_tile(lat, lon, zoom, "sentinel")
            if img is None:
                img = cls.fetch_satellite_tile(lat, lon, zoom, "nasa_gibs")

            if img is not None:
                images.append({"image": img, "source": "api", "location": loc})

        logger.info(f"Fetched {len(images)}/{count} satellite images from APIs")
        return images


# ================================================================
# SYNTHETIC SATELLITE IMAGE GENERATOR
# ================================================================

class SyntheticGenerator:
    """
    Generates high-quality synthetic satellite-like images with
    realistic terrain features and landslide annotations.
    """

    # Terrain color palettes — Indian biomes
    PALETTES = {
        "western_ghats": {
            # Dense tropical forests of Kerala, Kodaikanal, Coorg
            "vegetation": [(28, 95, 35), (35, 110, 30), (22, 85, 28), (40, 100, 38)],
            "bare": [(145, 115, 80), (155, 125, 85), (135, 105, 75)],
            "rock": [(150, 140, 125), (140, 130, 115), (160, 150, 135)],
        },
        "himalayan": {
            # Rocky high-altitude terrain of Uttarakhand, HP, Sikkim
            "vegetation": [(55, 80, 42), (65, 90, 48), (45, 70, 38)],
            "bare": [(170, 150, 115), (180, 160, 125), (160, 140, 108)],
            "rock": [(185, 180, 170), (170, 165, 155), (195, 190, 180)],
        },
        "nilgiris": {
            # Ooty, Coonoor — tea plantations and shola forests
            "vegetation": [(30, 105, 32), (38, 118, 28), (25, 95, 35), (45, 110, 40)],
            "bare": [(150, 120, 75), (160, 130, 80), (140, 110, 70)],
            "rock": [(140, 135, 125), (155, 150, 140), (130, 125, 115)],
        },
        "northeast": {
            # Meghalaya, Mizoram — lush monsoon forests
            "vegetation": [(20, 100, 25), (30, 115, 20), (18, 88, 30)],
            "bare": [(155, 115, 72), (165, 125, 78), (145, 105, 68)],
            "rock": [(138, 132, 122), (152, 146, 136), (128, 122, 112)],
        },
    }

    @classmethod
    def generate_terrain(cls, size=IMG_SIZE, palette_name=None, seed=None):
        """
        Generate a realistic satellite terrain image using Perlin-like noise.

        Args:
            size (int): Image size
            palette_name (str): Terrain palette name
            seed (int): Random seed

        Returns:
            PIL.Image: RGB terrain image
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if palette_name is None:
            palette_name = random.choice(list(cls.PALETTES.keys()))
        palette = cls.PALETTES.get(palette_name, cls.PALETTES["himalayan"])

        # Create base terrain using multi-scale noise
        img = np.zeros((size, size, 3), dtype=np.float32)

        # Generate multi-octave noise for realistic terrain
        for octave in range(5):
            freq = 2 ** octave
            amplitude = 1.0 / (octave + 1)
            noise = cls._generate_noise(size, freq * 4) * amplitude
            img[:, :, 0] += noise
            img[:, :, 1] += noise * 1.1  # Slightly more green
            img[:, :, 2] += noise * 0.9

        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Apply color palette
        result = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                val = img[y, x, 0]
                if val < 0.4:
                    # Vegetation
                    base = random.choice(palette["vegetation"])
                elif val < 0.7:
                    # Bare terrain
                    base = random.choice(palette["bare"])
                else:
                    # Rock
                    base = random.choice(palette["rock"])

                noise_r = random.randint(-15, 15)
                noise_g = random.randint(-15, 15)
                noise_b = random.randint(-15, 15)

                result[y, x] = [
                    max(0, min(255, base[0] + noise_r)),
                    max(0, min(255, base[1] + noise_g)),
                    max(0, min(255, base[2] + noise_b)),
                ]

        pil_img = Image.fromarray(result, "RGB")

        # Apply subtle blur for satellite look
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.8))

        # Add texture
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.3)

        return pil_img

    @staticmethod
    def _generate_noise(size, scale):
        """Generate smooth noise using interpolation."""
        small = np.random.randn(scale, scale)
        # Use PIL for smooth upscaling
        small_img = Image.fromarray(((small + 2) * 64).clip(0, 255).astype(np.uint8))
        large_img = small_img.resize((size, size), Image.BICUBIC)
        return np.array(large_img, dtype=np.float32) / 255.0

    @classmethod
    def add_landslide(cls, img, bbox=None):
        """
        Add a realistic landslide scar to the image.

        Args:
            img (PIL.Image): Input image
            bbox (tuple, optional): (x, y, w, h) for placement

        Returns:
            tuple: (modified PIL.Image, (x1, y1, x2, y2) normalized bbox)
        """
        w_img, h_img = img.size
        draw = ImageDraw.Draw(img)

        if bbox is None:
            # Random placement avoiding edges
            margin = int(w_img * 0.1)
            scar_w = random.randint(int(w_img * 0.08), int(w_img * 0.25))
            scar_h = random.randint(int(h_img * 0.12), int(h_img * 0.35))
            scar_x = random.randint(margin, w_img - scar_w - margin)
            scar_y = random.randint(margin, h_img - scar_h - margin)
        else:
            scar_x, scar_y, scar_w, scar_h = bbox

        # Landslide scar colors (bare earth, exposed soil)
        scar_colors = [
            (160, 140, 100), (170, 150, 110), (150, 130, 95),
            (175, 155, 115), (145, 125, 90), (165, 145, 105),
        ]

        # Draw irregular scar shape using ellipses and polygons
        pixels = np.array(img)
        for py in range(max(0, scar_y), min(h_img, scar_y + scar_h)):
            for px in range(max(0, scar_x), min(w_img, scar_x + scar_w)):
                # Create irregular elliptical boundary
                dx = (px - scar_x - scar_w / 2) / (scar_w / 2 + 0.1)
                dy = (py - scar_y - scar_h / 2) / (scar_h / 2 + 0.1)
                dist = dx * dx + dy * dy

                # Irregular boundary with noise
                threshold = 1.0 + 0.25 * math.sin(px * 0.1) * math.sin(py * 0.08)
                if dist < threshold:
                    base = random.choice(scar_colors)
                    # Edge blending
                    blend = max(0, 1.0 - dist * 0.7)
                    orig = pixels[py, px]
                    pixels[py, px] = [
                        int(orig[0] * (1 - blend) + base[0] * blend + random.randint(-8, 8)),
                        int(orig[1] * (1 - blend) + base[1] * blend + random.randint(-8, 8)),
                        int(orig[2] * (1 - blend) + base[2] * blend + random.randint(-8, 8)),
                    ]
                    pixels[py, px] = np.clip(pixels[py, px], 0, 255)

        # Add small rubble/scatter around scar edges
        for _ in range(random.randint(5, 15)):
            rx = scar_x + random.randint(-10, scar_w + 10)
            ry = scar_y + scar_h + random.randint(-5, 20)
            if 0 <= rx < w_img - 3 and 0 <= ry < h_img - 3:
                rs = random.randint(2, 5)
                rubble_color = random.choice(scar_colors)
                pixels[ry:min(ry + rs, h_img), rx:min(rx + rs, w_img)] = rubble_color

        modified = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

        # Compute YOLO normalized bbox
        x_center = (scar_x + scar_w / 2) / w_img
        y_center = (scar_y + scar_h / 2) / h_img
        norm_w = scar_w / w_img
        norm_h = scar_h / h_img

        return modified, (x_center, y_center, norm_w, norm_h)

    @classmethod
    def add_debris_flow(cls, img, start_pos=None):
        """
        Add a debris flow channel feature.

        Returns:
            tuple: (modified PIL.Image, (x_center, y_center, w, h) normalized bbox)
        """
        w_img, h_img = img.size
        pixels = np.array(img)

        if start_pos is None:
            sx = random.randint(w_img // 4, 3 * w_img // 4)
            sy = random.randint(h_img // 6, h_img // 2)
        else:
            sx, sy = start_pos

        debris_colors = [
            (130, 115, 85), (140, 125, 90), (120, 105, 80),
            (135, 120, 88), (125, 110, 82),
        ]

        min_x, min_y = sx, sy
        max_x, max_y = sx, sy

        cx, cy = sx, sy
        length = random.randint(60, 150)

        for i in range(length):
            cx += random.randint(-2, 3)
            cy += random.randint(1, 4)

            if 0 <= cx < w_img - 8 and 0 <= cy < h_img - 8:
                spread = max(2, random.randint(3, 8) - i // 25)
                color = random.choice(debris_colors)

                for dy in range(spread):
                    for dx in range(spread):
                        if cy + dy < h_img and cx + dx < w_img:
                            blend = 0.7 + random.uniform(-0.1, 0.1)
                            orig = pixels[cy + dy, cx + dx]
                            pixels[cy + dy, cx + dx] = np.clip([
                                int(orig[0] * (1 - blend) + color[0] * blend),
                                int(orig[1] * (1 - blend) + color[1] * blend),
                                int(orig[2] * (1 - blend) + color[2] * blend),
                            ], 0, 255)

                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx + spread)
                max_y = max(max_y, cy + spread)

        modified = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

        # YOLO bbox
        bw = max_x - min_x
        bh = max_y - min_y
        x_center = (min_x + bw / 2) / w_img
        y_center = (min_y + bh / 2) / h_img
        norm_w = bw / w_img
        norm_h = bh / h_img

        return modified, (x_center, y_center, norm_w, norm_h)

    @classmethod
    def add_normal_terrain_feature(cls, img, feature_type=None):
        """
        Add a normal terrain feature (road, river, vegetation patch).

        Returns:
            tuple: (modified PIL.Image, (x_center, y_center, w, h) normalized bbox)
        """
        w_img, h_img = img.size
        pixels = np.array(img)

        if feature_type is None:
            feature_type = random.choice(["vegetation_patch", "rocky_area", "water_body"])

        margin = int(w_img * 0.1)
        feat_w = random.randint(int(w_img * 0.06), int(w_img * 0.15))
        feat_h = random.randint(int(h_img * 0.06), int(h_img * 0.15))
        feat_x = random.randint(margin, w_img - feat_w - margin)
        feat_y = random.randint(margin, h_img - feat_h - margin)

        if feature_type == "vegetation_patch":
            colors = [(30, 100, 35), (25, 90, 30), (35, 110, 40)]
        elif feature_type == "rocky_area":
            colors = [(170, 165, 155), (160, 155, 145), (180, 175, 165)]
        else:  # water_body
            colors = [(60, 80, 130), (50, 70, 120), (55, 75, 125)]

        for py in range(max(0, feat_y), min(h_img, feat_y + feat_h)):
            for px in range(max(0, feat_x), min(w_img, feat_x + feat_w)):
                dx = (px - feat_x - feat_w / 2) / (feat_w / 2 + 0.1)
                dy = (py - feat_y - feat_h / 2) / (feat_h / 2 + 0.1)
                if dx * dx + dy * dy < 1.2:
                    color = random.choice(colors)
                    blend = 0.6
                    orig = pixels[py, px]
                    pixels[py, px] = np.clip([
                        int(orig[0] * (1 - blend) + color[0] * blend),
                        int(orig[1] * (1 - blend) + color[1] * blend),
                        int(orig[2] * (1 - blend) + color[2] * blend),
                    ], 0, 255)

        modified = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

        x_center = (feat_x + feat_w / 2) / w_img
        y_center = (feat_y + feat_h / 2) / h_img
        norm_w = feat_w / w_img
        norm_h = feat_h / h_img

        return modified, (x_center, y_center, norm_w, norm_h)


# ================================================================
# AUGMENTATION
# ================================================================

def augment_image(img):
    """Apply random augmentations to a satellite image."""
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.85, 1.15))

    # Random color shift
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.9, 1.1))

    # Random rotation (0, 90, 180, 270)
    if random.random() > 0.5:
        angle = random.choice([90, 180, 270])
        img = img.rotate(angle, expand=False)

    # Random horizontal/vertical flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return img


def overlay_landslide_on_real(real_img, generator):
    """
    Overlay synthetic landslide features onto a real satellite image.

    Args:
        real_img (PIL.Image): Real satellite image
        generator (SyntheticGenerator): Generator for features

    Returns:
        tuple: (annotated image, list of (class_id, x, y, w, h))
    """
    img = real_img.copy().resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    annotations = []

    # Add 1-3 landslide features
    num_landslides = random.randint(1, 3)
    for _ in range(num_landslides):
        img, bbox = generator.add_landslide(img)
        annotations.append((0, *bbox))

    # Optionally add debris flow
    if random.random() > 0.4:
        img, bbox = generator.add_debris_flow(img)
        annotations.append((1, *bbox))

    # Optionally add normal terrain markers
    if random.random() > 0.5:
        img, bbox = generator.add_normal_terrain_feature(img)
        annotations.append((2, *bbox))

    return img, annotations


# ================================================================
# DATASET GENERATION
# ================================================================

def generate_dataset(output_dir, num_train=120, num_val=30, num_test=20,
                     use_api=True, seed=42):
    """
    Generate a complete YOLO-format dataset.

    Args:
        output_dir (str): Dataset root directory
        num_train (int): Number of training images
        num_val (int): Number of validation images
        num_test (int): Number of test images
        use_api (bool): Whether to try fetching real satellite images
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    total = num_train + num_val + num_test
    gen = SyntheticGenerator()

    logger.info("=" * 60)
    logger.info("  DATASET GENERATION FOR LANDSLIDE DETECTION")
    logger.info("=" * 60)
    logger.info(f"  Train: {num_train} | Val: {num_val} | Test: {num_test}")
    logger.info(f"  Total: {total} images")
    logger.info(f"  API Fetch: {'Enabled' if use_api else 'Disabled'}")
    logger.info("=" * 60)

    # Create directories
    splits = {"train": num_train, "val": num_val, "test": num_test}
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # ---- Step 1: Fetch real satellite images from APIs ----
    api_images = []
    if use_api and HAS_REQUESTS:
        logger.info("\n📡 Fetching satellite images from NASA & Sentinel APIs...")
        api_count = min(total // 3, 30)  # Use API for up to 1/3 of images
        try:
            api_images = NASAImageFetcher.fetch_batch(count=api_count, zoom=12)
            logger.info(f"✅ Retrieved {len(api_images)} real satellite images from APIs")
        except Exception as e:
            logger.warning(f"⚠ API fetch encountered error: {e}")
            logger.info("  Continuing with synthetic generation...")

    # ---- Step 2: Generate dataset ----
    image_idx = 0
    api_idx = 0

    for split, count in splits.items():
        logger.info(f"\n🔧 Generating {split} split ({count} images)...")
        images_dir = os.path.join(output_dir, split, "images")
        labels_dir = os.path.join(output_dir, split, "labels")

        for i in range(count):
            image_idx += 1
            filename = f"satellite_{image_idx:04d}"

            # Decide image source
            use_real = (api_idx < len(api_images)) and random.random() > 0.3
            annotations = []

            if use_real:
                # Use real satellite image with synthetic landslide overlay
                real_data = api_images[api_idx]
                api_idx += 1
                img, annotations = overlay_landslide_on_real(real_data["image"], gen)
                src = f"API ({real_data.get('location', {}).get('name', 'unknown')})"
            else:
                # Generate fully synthetic image
                palette = random.choice(list(SyntheticGenerator.PALETTES.keys()))
                img = gen.generate_terrain(
                    size=IMG_SIZE,
                    palette_name=palette,
                    seed=seed + image_idx
                )

                # Add landslide features
                scenario = random.choice([
                    "landslide_only",
                    "landslide_debris",
                    "mixed",
                    "multiple_landslides",
                ])

                if scenario == "landslide_only":
                    img, bbox = gen.add_landslide(img)
                    annotations.append((0, *bbox))

                elif scenario == "landslide_debris":
                    img, bbox = gen.add_landslide(img)
                    annotations.append((0, *bbox))
                    img, bbox = gen.add_debris_flow(img)
                    annotations.append((1, *bbox))

                elif scenario == "mixed":
                    img, bbox = gen.add_landslide(img)
                    annotations.append((0, *bbox))
                    if random.random() > 0.5:
                        img, bbox = gen.add_debris_flow(img)
                        annotations.append((1, *bbox))
                    img, bbox = gen.add_normal_terrain_feature(img)
                    annotations.append((2, *bbox))

                elif scenario == "multiple_landslides":
                    for _ in range(random.randint(2, 3)):
                        img, bbox = gen.add_landslide(img)
                        annotations.append((0, *bbox))
                    if random.random() > 0.5:
                        img, bbox = gen.add_debris_flow(img)
                        annotations.append((1, *bbox))

                src = f"synthetic ({palette})"

            # Apply augmentation
            if split == "train" and random.random() > 0.3:
                img = augment_image(img)

            # Save image
            img_path = os.path.join(images_dir, filename + ".jpg")
            img.save(img_path, "JPEG", quality=95)

            # Save YOLO annotations
            label_path = os.path.join(labels_dir, filename + ".txt")
            with open(label_path, "w") as f:
                for ann in annotations:
                    cls_id, xc, yc, w, h = ann
                    # Clamp values to [0, 1]
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    w = max(0.001, min(1.0, w))
                    h = max(0.001, min(1.0, h))
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"  [{split}] {i+1}/{count} - {filename} ({src})")

    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("  DATASET GENERATION COMPLETE")
    logger.info("=" * 60)

    for split in splits:
        img_count = len(os.listdir(os.path.join(output_dir, split, "images")))
        lbl_count = len(os.listdir(os.path.join(output_dir, split, "labels")))
        logger.info(f"  {split:5s}: {img_count} images, {lbl_count} labels")

    logger.info(f"\n  API images used: {api_idx}")
    logger.info(f"  Synthetic images: {total - api_idx}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate landslide detection dataset"
    )
    parser.add_argument("--output", type=str, default="dataset",
                        help="Output directory (default: dataset)")
    parser.add_argument("--train", type=int, default=120,
                        help="Number of training images (default: 120)")
    parser.add_argument("--val", type=int, default=30,
                        help="Number of validation images (default: 30)")
    parser.add_argument("--test", type=int, default=20,
                        help="Number of test images (default: 20)")
    parser.add_argument("--no-api", action="store_true",
                        help="Disable API fetch, use only synthetic images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        num_train=args.train,
        num_val=args.val,
        num_test=args.test,
        use_api=not args.no_api,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
