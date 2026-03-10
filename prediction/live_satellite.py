"""
Live Satellite Imagery Fetcher
==============================
Fetches real satellite imagery tiles based on latitude and longitude
using the public Esri World Imagery map service. No API keys required.
"""

import math
import logging
import numpy as np
import requests
from PIL import Image
import io

logger = logging.getLogger(__name__)

class LiveSatelliteFetcher:
    # Public Esri World Imagery basemap
    TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    @staticmethod
    def lat_lon_to_tile(lat, lon, zoom):
        """Convert latitude and longitude to Slippy Map XY tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x_tile, y_tile

    @staticmethod
    def fetch_live_image(lat, lon, zoom=14, grid_size=3):
        """
        Fetch a stitched grid of satellite tiles surrounding the target lat/lon.

        Args:
            lat (float): Latitude
            lon (float): Longitude
            zoom (int): Zoom level (14 provides good regional context)
            grid_size (int): Must be odd (e.g., 3 means a 3x3 grid of 256x256 tiles)

        Returns:
            np.ndarray: Stitched BGR image (H, W, 3)
        """
        center_x, center_y = LiveSatelliteFetcher.lat_lon_to_tile(lat, lon, zoom)
        
        offset = grid_size // 2
        tiles = []
        
        # Tile size is exactly 256x256
        for dy in range(-offset, offset + 1):
            row_tiles = []
            for dx in range(-offset, offset + 1):
                x = center_x + dx
                y = center_y + dy
                url = LiveSatelliteFetcher.TILE_URL.format(z=zoom, y=y, x=x)
                
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    row_tiles.append(np.array(img))
                except Exception as e:
                    logger.warning(f"Failed to fetch tile at ({x}, {y}): {e}")
                    # Fallback black tile
                    row_tiles.append(np.zeros((256, 256, 3), dtype=np.uint8))
            
            # Horizontally concatenate the row
            if row_tiles:
                tiles.append(np.concatenate(row_tiles, axis=1))
        
        # Vertically concatenate all rows
        if tiles:
            stitched_rgb = np.concatenate(tiles, axis=0)
            # Convert RGB to BGR for OpenCV/YOLO
            return stitched_rgb[:, :, ::-1]
        
        return None
