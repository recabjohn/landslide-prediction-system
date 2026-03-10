"""
Satellite Image Fetcher Module
===============================
Fetches satellite imagery from various sources:
  1. Sentinel-2 via Copernicus Data Space API
  2. Local file system
  3. Demo mode (procedurally generated samples)

This module provides a unified interface for acquiring satellite
images regardless of the source.
"""

import os
import logging
import numpy as np
from PIL import Image

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Configure logging
logger = logging.getLogger(__name__)


class SatelliteFetcher:
    """
    Unified satellite image fetcher with graceful fallback chain:
    Sentinel-2 API → Local Files → Demo Images
    """

    # Sentinel-2 Copernicus Data Space API endpoint
    SENTINEL_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

    def __init__(self, api_client_id=None, api_client_secret=None):
        """
        Initialize the satellite fetcher.

        Args:
            api_client_id (str, optional): Copernicus OAuth client ID
            api_client_secret (str, optional): Copernicus OAuth client secret
        """
        self.api_client_id = api_client_id or os.environ.get("SENTINEL_CLIENT_ID")
        self.api_client_secret = api_client_secret or os.environ.get("SENTINEL_CLIENT_SECRET")
        self._access_token = None

    # ----------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------

    def fetch(self, lat=None, lon=None, local_path=None, demo=False):
        """
        Fetch a satellite image using the fallback chain.

        Priority:
          1. If demo=True → return a demo image
          2. If local_path is provided → load from disk
          3. If lat/lon are provided and API credentials exist → Sentinel-2
          4. Fallback to demo image

        Args:
            lat (float, optional): Latitude of the region of interest
            lon (float, optional): Longitude of the region of interest
            local_path (str, optional): Path to a local satellite image
            demo (bool): Force demo mode

        Returns:
            np.ndarray: BGR image as a NumPy array (H, W, 3)
            str: Source description string
        """
        # 1. Demo mode
        if demo:
            logger.info("Demo mode: generating sample satellite image")
            return self.fetch_demo_image(), "demo"

        # 2. Local file
        if local_path and os.path.isfile(local_path):
            logger.info(f"Loading local image: {local_path}")
            return self.fetch_from_local(local_path), f"local:{local_path}"

        # 3. Sentinel-2 API
        if lat is not None and lon is not None and self.api_client_id:
            try:
                logger.info(f"Fetching Sentinel-2 image at ({lat}, {lon})")
                img = self.fetch_sentinel2(lat, lon)
                if img is not None:
                    return img, f"sentinel2:({lat},{lon})"
            except Exception as e:
                logger.warning(f"Sentinel-2 fetch failed: {e}. Falling back to demo.")

        # 4. Fallback
        logger.info("Falling back to demo image")
        return self.fetch_demo_image(), "demo"

    # ----------------------------------------------------------------
    # SENTINEL-2 FETCHER
    # ----------------------------------------------------------------

    def _authenticate(self):
        """Obtain an OAuth2 access token from Copernicus Data Space."""
        if not HAS_REQUESTS:
            raise RuntimeError("'requests' library is required for Sentinel-2 API")

        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        response = requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": self.api_client_id,
            "client_secret": self.api_client_secret,
        }, timeout=30)
        response.raise_for_status()
        self._access_token = response.json()["access_token"]
        logger.info("Sentinel-2 authentication successful")

    def fetch_sentinel2(self, lat, lon, resolution=10, size=640):
        """
        Fetch a Sentinel-2 true-color image from Copernicus Data Space.

        Args:
            lat (float): Latitude (WGS84)
            lon (float): Longitude (WGS84)
            resolution (int): Spatial resolution in meters (default 10m)
            size (int): Output image size in pixels (default 640)

        Returns:
            np.ndarray: BGR image (H, W, 3) or None on failure
        """
        if not self._access_token:
            self._authenticate()

        # Compute bounding box (~6.4 km at 10 m/px for 640 px)
        offset = (size * resolution) / 111320.0 / 2  # approx degrees
        bbox = [lon - offset, lat - offset, lon + offset, lat + offset]

        # Sentinel Hub Process API evalscript for true color
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: [{ bands: ["B04", "B03", "B02"] }],
            output: { bands: 3, sampleType: "AUTO" }
          };
        }
        function evaluatePixel(sample) {
          return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
        }
        """

        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "maxCloudCoverage": 20,
                        "timeRange": {
                            "from": "2024-01-01T00:00:00Z",
                            "to": "2024-12-31T23:59:59Z"
                        }
                    }
                }]
            },
            "output": {
                "width": size,
                "height": size,
                "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
            },
            "evalscript": evalscript
        }

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Accept": "image/png"
        }

        response = requests.post(self.SENTINEL_API_URL, json=payload,
                                 headers=headers, timeout=60)
        response.raise_for_status()

        # Decode PNG response
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = Image.open(__import__('io').BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)

        # Convert RGB → BGR for OpenCV compatibility
        img_bgr = img_np[:, :, ::-1].copy()
        logger.info(f"Sentinel-2 image fetched: shape={img_bgr.shape}")
        return img_bgr

    # ----------------------------------------------------------------
    # LOCAL FILE LOADER
    # ----------------------------------------------------------------

    def fetch_from_local(self, path):
        """
        Load a satellite image from local file system.

        Args:
            path (str): Absolute or relative path to the image file

        Returns:
            np.ndarray: BGR image (H, W, 3)

        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        img_bgr = img_np[:, :, ::-1].copy()  # RGB → BGR
        logger.info(f"Local image loaded: {path}, shape={img_bgr.shape}")
        return img_bgr

    # ----------------------------------------------------------------
    # DEMO IMAGE GENERATOR
    # ----------------------------------------------------------------

    def fetch_demo_image(self, size=640, seed=None):
        """
        Generate a procedural satellite-like demo image with terrain features.

        Creates a realistic-looking terrain image with:
          - Base terrain with varying elevation colors
          - Mountain ridges and valleys
          - Vegetation-like textures
          - Simulated landslide scar regions

        Args:
            size (int): Image size (width = height)
            seed (int, optional): Random seed for reproducibility

        Returns:
            np.ndarray: BGR image (size, size, 3)
        """
        if seed is not None:
            np.random.seed(seed)

        img = np.zeros((size, size, 3), dtype=np.uint8)

        # --- Base terrain gradient (greenish-brown) ---
        for y in range(size):
            ratio = y / size
            # Green channel: lush at top, brown at bottom
            g = int(80 + 60 * (1 - ratio) + np.random.randint(-10, 10))
            # Red channel: earth tones
            r = int(60 + 80 * ratio + np.random.randint(-10, 10))
            # Blue channel: subtle sky reflection
            b = int(40 + 20 * (1 - ratio) + np.random.randint(-5, 5))
            img[y, :] = [
                np.clip(b, 0, 255),
                np.clip(g, 0, 255),
                np.clip(r, 0, 255)
            ]

        # --- Add noise for natural texture ---
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # --- Add mountain ridges ---
        for _ in range(3):
            ridge_y = np.random.randint(size // 4, 3 * size // 4)
            for x in range(size):
                y_offset = int(30 * np.sin(x / 40 + np.random.random()))
                y_pos = ridge_y + y_offset
                if 0 <= y_pos < size - 5:
                    # Dark ridge shadow
                    img[y_pos:y_pos + 3, x] = [40, 50, 45]
                    # Light ridge highlight
                    if y_pos > 2:
                        img[y_pos - 2:y_pos, x] = [80, 120, 100]

        # --- Add landslide scar (brownish-grey bare earth) ---
        scar_x = np.random.randint(size // 4, size // 2)
        scar_y = np.random.randint(size // 4, size // 2)
        scar_w = np.random.randint(80, 160)
        scar_h = np.random.randint(100, 200)

        for y in range(max(0, scar_y), min(size, scar_y + scar_h)):
            for x in range(max(0, scar_x), min(size, scar_x + scar_w)):
                # Irregular scar shape
                dist = ((x - scar_x - scar_w // 2) ** 2 / (scar_w // 2 + 1) ** 2 +
                        (y - scar_y - scar_h // 2) ** 2 / (scar_h // 2 + 1) ** 2)
                if dist < 1.0 + 0.3 * np.random.random():
                    # Bare earth colors (grey-brown)
                    img[y, x] = [
                        np.clip(int(90 + np.random.randint(-20, 20)), 0, 255),
                        np.clip(int(100 + np.random.randint(-20, 20)), 0, 255),
                        np.clip(int(120 + np.random.randint(-15, 15)), 0, 255)
                    ]

        # --- Add a second smaller debris flow ---
        debris_x = np.random.randint(size // 2, 3 * size // 4)
        debris_y = np.random.randint(size // 3, 2 * size // 3)
        for i in range(80):
            dx = debris_x + int(i * 0.8 + np.random.randint(-5, 5))
            dy = debris_y + int(i * 1.5 + np.random.randint(-3, 3))
            if 0 <= dx < size - 6 and 0 <= dy < size - 6:
                spread = max(3, 8 - i // 15)
                img[dy:dy + spread, dx:dx + spread] = [
                    np.clip(int(70 + np.random.randint(-10, 10)), 0, 255),
                    np.clip(int(80 + np.random.randint(-10, 10)), 0, 255),
                    np.clip(int(95 + np.random.randint(-10, 10)), 0, 255)
                ]

        # --- Add vegetation patches ---
        for _ in range(20):
            vx = np.random.randint(0, size - 20)
            vy = np.random.randint(0, size - 20)
            vs = np.random.randint(5, 20)
            green = np.random.randint(90, 150)
            img[vy:vy + vs, vx:vx + vs, 1] = np.clip(
                img[vy:vy + vs, vx:vx + vs, 1].astype(np.int16) + green // 3,
                0, 255
            ).astype(np.uint8)

        logger.info(f"Demo satellite image generated: shape={img.shape}")
        return img

    # ----------------------------------------------------------------
    # UTILITY
    # ----------------------------------------------------------------

    def save_image(self, image, path):
        """Save a BGR numpy image to disk."""
        img_rgb = image[:, :, ::-1]  # BGR → RGB
        Image.fromarray(img_rgb).save(path)
        logger.info(f"Image saved to {path}")

    @staticmethod
    def list_local_images(directory, extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        """List all image files in a directory."""
        if not os.path.isdir(directory):
            return []
        return [
            os.path.join(directory, f)
            for f in sorted(os.listdir(directory))
            if f.lower().endswith(extensions)
        ]
