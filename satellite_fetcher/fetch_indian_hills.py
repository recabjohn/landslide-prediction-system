"""
Indian Hills Satellite Image Fetcher
======================================
Downloads real satellite imagery tiles from Esri World Imagery
for known landslide-prone Indian hill stations.

No API key required — uses the public Esri basemap tile service.

Usage:
    python satellite_fetcher/fetch_indian_hills.py
"""

import os
import math
import logging
import numpy as np

try:
    import requests
    from PIL import Image
    import io
except ImportError as e:
    raise ImportError(f"Required packages missing: {e}. Install with: pip install requests Pillow")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ================================================================
# KNOWN LANDSLIDE-PRONE INDIAN HILL LOCATIONS
# ================================================================

INDIAN_HILLS = {
    "kedarnath_uttarakhand": {
        "lat": 30.7352, "lon": 79.0669,
        "label": "Kedarnath, Uttarakhand",
        "region": "Himalayas",
        "notes": "2013 catastrophic landslide & flash flood site",
    },
    "joshimath_uttarakhand": {
        "lat": 30.5550, "lon": 79.5650,
        "label": "Joshimath, Uttarakhand",
        "region": "Himalayas",
        "notes": "Active land subsidence zone since 2022",
    },
    "munnar_kerala": {
        "lat": 10.0889, "lon": 77.0595,
        "label": "Munnar, Kerala",
        "region": "Western Ghats",
        "notes": "Pettimudi landslide area (2020)",
    },
    "wayanad_kerala": {
        "lat": 11.6854, "lon": 76.1320,
        "label": "Wayanad, Kerala",
        "region": "Western Ghats",
        "notes": "Puthumala / Kavalappara landslide zone (2019)",
    },
    "kodaikanal_tamilnadu": {
        "lat": 10.2381, "lon": 77.4892,
        "label": "Kodaikanal, Tamil Nadu",
        "region": "Western Ghats",
        "notes": "Steep terrain, monsoon-triggered slides",
    },
    "shimla_himachal": {
        "lat": 31.1048, "lon": 77.1734,
        "label": "Shimla, Himachal Pradesh",
        "region": "Himalayas",
        "notes": "NH-5 landslide corridor",
    },
    "darjeeling_westbengal": {
        "lat": 27.0360, "lon": 88.2627,
        "label": "Darjeeling, West Bengal",
        "region": "Himalayas",
        "notes": "Tea garden slope instability",
    },
    "cherrapunji_meghalaya": {
        "lat": 25.2700, "lon": 91.7200,
        "label": "Cherrapunji, Meghalaya",
        "region": "Northeast India",
        "notes": "Wettest place on Earth, extreme erosion",
    },
    "mahabaleshwar_maharashtra": {
        "lat": 17.9307, "lon": 73.6477,
        "label": "Mahabaleshwar, Maharashtra",
        "region": "Western Ghats",
        "notes": "Monsoon debris flows on laterite",
    },
    "manali_himachal": {
        "lat": 32.2396, "lon": 77.1887,
        "label": "Manali, Himachal Pradesh",
        "region": "Himalayas",
        "notes": "Beas River valley landslide zone",
    },
    "idukki_kerala": {
        "lat": 9.8494, "lon": 76.9710,
        "label": "Idukki, Kerala",
        "region": "Western Ghats",
        "notes": "2018 floods, major landslide damage",
    },
    "gangtok_sikkim": {
        "lat": 27.3389, "lon": 88.6065,
        "label": "Gangtok, Sikkim",
        "region": "Himalayas",
        "notes": "South Lhonak Lake glacial outburst risk",
    },
    "pithoragarh_uttarakhand": {
        "lat": 29.5829, "lon": 80.2182,
        "label": "Pithoragarh, Uttarakhand",
        "region": "Himalayas",
        "notes": "Nepal border, active seismic zone",
    },
    "coorg_karnataka": {
        "lat": 12.3375, "lon": 75.8069,
        "label": "Coorg, Karnataka",
        "region": "Western Ghats",
        "notes": "Kodagu floods and landslides (2018)",
    },
    "nilgiris_tamilnadu": {
        "lat": 11.4102, "lon": 76.6950,
        "label": "Nilgiris (Ooty), Tamil Nadu",
        "region": "Western Ghats",
        "notes": "Coonoor-Ooty road landslide corridor",
    },
    # --- Additional South Indian Hills ---
    "yelagiri_tamilnadu": {
        "lat": 12.5809, "lon": 78.6374,
        "label": "Yelagiri Hills, Tamil Nadu",
        "region": "Eastern Ghats",
        "notes": "Steep ghat road, monsoon slope failures",
    },
    "yercaud_tamilnadu": {
        "lat": 11.7754, "lon": 78.2050,
        "label": "Yercaud, Tamil Nadu",
        "region": "Eastern Ghats (Shevaroy Hills)",
        "notes": "Shevaroy Hills, hairpin bend landslides",
    },
    "valparai_tamilnadu": {
        "lat": 10.3264, "lon": 76.9545,
        "label": "Valparai, Tamil Nadu",
        "region": "Western Ghats (Anamalai)",
        "notes": "40-hairpin road, heavy monsoon slides",
    },
    "coonoor_tamilnadu": {
        "lat": 11.3530, "lon": 76.7959,
        "label": "Coonoor, Tamil Nadu",
        "region": "Nilgiri Hills",
        "notes": "2009 Nilgiris landslide disaster zone",
    },
    "vagamon_kerala": {
        "lat": 9.6862, "lon": 76.9005,
        "label": "Vagamon, Kerala",
        "region": "Western Ghats",
        "notes": "Steep tea estate slopes, monsoon erosion",
    },
    "ponmudi_kerala": {
        "lat": 8.7564, "lon": 77.1133,
        "label": "Ponmudi, Kerala",
        "region": "Western Ghats",
        "notes": "22 hairpin bends, Trivandrum hill station",
    },
    "agumbe_karnataka": {
        "lat": 13.5027, "lon": 75.0925,
        "label": "Agumbe, Karnataka",
        "region": "Western Ghats",
        "notes": "Cherrapunji of the South, extreme rainfall",
    },
    "chikmagalur_karnataka": {
        "lat": 13.3161, "lon": 75.7720,
        "label": "Chikmagalur, Karnataka",
        "region": "Western Ghats",
        "notes": "Mullayanagiri range, coffee estate slides",
    },
    "kolli_hills_tamilnadu": {
        "lat": 11.2497, "lon": 78.3555,
        "label": "Kolli Hills, Tamil Nadu",
        "region": "Eastern Ghats",
        "notes": "70 hairpin bends, remote hill terrain",
    },
    "anamalai_tamilnadu": {
        "lat": 10.3500, "lon": 76.8300,
        "label": "Anamalai Hills, Tamil Nadu",
        "region": "Western Ghats (Anamalai Tiger Reserve)",
        "notes": "Biodiversity hotspot, steep forested slopes",
    },
    # --- Additional High-Risk Himalayan Zones ---
    "badrinath_uttarakhand": {
        "lat": 30.7433, "lon": 79.4938,
        "label": "Badrinath, Uttarakhand",
        "region": "Himalayas",
        "notes": "Alaknanda Valley, flash flood prone pilgrimage route",
    },
    "rudraprayag_uttarakhand": {
        "lat": 30.2850, "lon": 78.9831,
        "label": "Rudraprayag, Uttarakhand",
        "region": "Himalayas",
        "notes": "2013 Kedarnath flood confluence zone",
    },
    "nainital_uttarakhand": {
        "lat": 29.3919, "lon": 79.4542,
        "label": "Nainital, Uttarakhand",
        "region": "Kumaon Hills",
        "notes": "Lake town, fragile slope geology",
    },
    "mussoorie_uttarakhand": {
        "lat": 30.4598, "lon": 78.0644,
        "label": "Mussoorie, Uttarakhand",
        "region": "Himalayas",
        "notes": "Queen of Hills, monsoon landslide corridor",
    },
    "tehri_uttarakhand": {
        "lat": 30.3781, "lon": 78.4836,
        "label": "Tehri, Uttarakhand",
        "region": "Himalayas",
        "notes": "Dam zone, reservoir-induced slope instability",
    },
    "kullu_himachal": {
        "lat": 31.9574, "lon": 77.1089,
        "label": "Kullu, Himachal Pradesh",
        "region": "Himalayas",
        "notes": "Beas Valley, recurring monsoon floods",
    },
    "kinnaur_himachal": {
        "lat": 31.5800, "lon": 78.4700,
        "label": "Kinnaur, Himachal Pradesh",
        "region": "Himalayas",
        "notes": "NH-5 landslide corridor, 2021 disaster zone",
    },
    "lahaul_spiti_himachal": {
        "lat": 32.5500, "lon": 77.5000,
        "label": "Lahaul-Spiti, Himachal Pradesh",
        "region": "Trans-Himalaya",
        "notes": "High-altitude desert, glacial lake outburst risk",
    },
    "dalhousie_himachal": {
        "lat": 32.5387, "lon": 75.9707,
        "label": "Dalhousie, Himachal Pradesh",
        "region": "Dhauladhar Range",
        "notes": "Colonial hill station, steep pine slopes",
    },
    "kasauli_himachal": {
        "lat": 30.8988, "lon": 76.9654,
        "label": "Kasauli, Himachal Pradesh",
        "region": "Shivalik Hills",
        "notes": "Lower Himalayan cantonment, monsoon slides",
    },
    # --- Northeast India Expansion ---
    "tawang_arunachal": {
        "lat": 27.5860, "lon": 91.8594,
        "label": "Tawang, Arunachal Pradesh",
        "region": "Eastern Himalayas",
        "notes": "Strategic high-altitude zone, seismic activity",
    },
    "ziro_arunachal": {
        "lat": 27.5455, "lon": 93.8252,
        "label": "Ziro Valley, Arunachal Pradesh",
        "region": "Eastern Himalayas",
        "notes": "UNESCO World Heritage Site, monsoon erosion",
    },
    "mawsynram_meghalaya": {
        "lat": 25.2971, "lon": 91.5821,
        "label": "Mawsynram, Meghalaya",
        "region": "Northeast India",
        "notes": "Wettest place on Earth, extreme rainfall erosion",
    },
    "kohima_nagaland": {
        "lat": 25.6751, "lon": 94.1086,
        "label": "Kohima, Nagaland",
        "region": "Naga Hills",
        "notes": "NH-29 landslide corridor, WWII memorial town",
    },
    "imphal_manipur": {
        "lat": 24.8170, "lon": 93.9368,
        "label": "Imphal, Manipur",
        "region": "Manipur Valley",
        "notes": "NH-37/39 landslide zones, monsoon flooding",
    },
    # --- Additional Western Ghats ---
    "kudremukh_karnataka": {
        "lat": 13.2253, "lon": 75.2500,
        "label": "Kudremukh, Karnataka",
        "region": "Western Ghats",
        "notes": "Highest peak in Karnataka, mining-affected slopes",
    },
    "kemmanagundi_karnataka": {
        "lat": 13.5322, "lon": 75.7450,
        "label": "Kemmanagundi, Karnataka",
        "region": "Baba Budan Hills",
        "notes": "Coffee estate landslide zones",
    },
    "silent_valley_kerala": {
        "lat": 11.0790, "lon": 76.4350,
        "label": "Silent Valley, Kerala",
        "region": "Western Ghats",
        "notes": "Dense rainforest, steep biodiversity hotspot",
    },
    "palani_tamilnadu": {
        "lat": 10.4391, "lon": 77.5206,
        "label": "Palani Hills, Tamil Nadu",
        "region": "Western Ghats",
        "notes": "Temple town, steep ghat road landslides",
    },
    "meghamalai_tamilnadu": {
        "lat": 9.6500, "lon": 77.4000,
        "label": "Meghamalai, Tamil Nadu",
        "region": "Western Ghats",
        "notes": "Cloud Mountain, remote high-altitude slopes",
    },
    "javadi_hills_tamilnadu": {
        "lat": 12.4500, "lon": 78.8500,
        "label": "Javadi Hills, Tamil Nadu",
        "region": "Eastern Ghats",
        "notes": "Tribal hill area, monsoon slope failures",
    },
    # --- Kashmir & Ladakh ---
    "gulmarg_kashmir": {
        "lat": 34.0500, "lon": 74.3800,
        "label": "Gulmarg, Kashmir",
        "region": "Pir Panjal Range",
        "notes": "Ski resort, avalanche and landslide risk",
    },
    "pahalgam_kashmir": {
        "lat": 34.0161, "lon": 75.3150,
        "label": "Pahalgam, Kashmir",
        "region": "Lidder Valley",
        "notes": "Amarnath pilgrimage base, cloud burst zone",
    },
    "sonamarg_kashmir": {
        "lat": 34.3039, "lon": 75.2938,
        "label": "Sonamarg, Kashmir",
        "region": "Sindh Valley",
        "notes": "Zoji La corridor, extreme avalanche zone",
    },
    "leh_ladakh": {
        "lat": 34.1526, "lon": 77.5771,
        "label": "Leh, Ladakh",
        "region": "Trans-Himalaya",
        "notes": "Flash flood zone, 2010 cloudburst disaster site",
    },
    "kargil_ladakh": {
        "lat": 34.5539, "lon": 76.1349,
        "label": "Kargil, Ladakh",
        "region": "Trans-Himalaya",
        "notes": "Strategic area, glacial lake outburst risk",
    },
}


# ================================================================
# TILE FETCHING
# ================================================================

TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def lat_lon_to_tile(lat, lon, zoom):
    """Convert latitude/longitude to Slippy Map tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x_tile, y_tile


def fetch_stitched_image(lat, lon, zoom=14, grid_size=3):
    """
    Fetch and stitch a grid of satellite tiles for a given location.

    Args:
        lat: Latitude
        lon: Longitude
        zoom: Tile zoom level (14 = good regional context)
        grid_size: Grid dimension (3 = 3×3 = 768×768 px)

    Returns:
        PIL.Image or None
    """
    center_x, center_y = lat_lon_to_tile(lat, lon, zoom)
    offset = grid_size // 2
    tiles = []

    for dy in range(-offset, offset + 1):
        row_tiles = []
        for dx in range(-offset, offset + 1):
            x = center_x + dx
            y = center_y + dy
            url = TILE_URL.format(z=zoom, y=y, x=x)

            try:
                resp = requests.get(url, timeout=15, headers={
                    "User-Agent": "LandslideMonitor/1.0 (Academic Research)"
                })
                resp.raise_for_status()
                tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
                row_tiles.append(np.array(tile))
            except Exception as e:
                logger.warning(f"  Failed tile ({x},{y}): {e}")
                row_tiles.append(np.zeros((256, 256, 3), dtype=np.uint8))

        tiles.append(np.concatenate(row_tiles, axis=1))

    stitched = np.concatenate(tiles, axis=0)
    return Image.fromarray(stitched)


def fetch_all_indian_hills(output_dir=None, zoom=14, grid_size=3):
    """
    Download satellite images for all listed Indian hill stations.

    Args:
        output_dir: Output directory (default: dataset/indian_hills/)
        zoom: Tile zoom level
        grid_size: Grid dimension per location

    Returns:
        list of saved file paths
    """
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "dataset", "indian_hills")

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    logger.info(f"Downloading satellite images for {len(INDIAN_HILLS)} Indian hill stations...")
    logger.info(f"Output directory: {output_dir}")

    for name, info in INDIAN_HILLS.items():
        filename = f"{name}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Skip if already downloaded
        if os.path.isfile(filepath):
            logger.info(f"  ✓ {info['label']}: already exists, skipping")
            saved.append(filepath)
            continue

        logger.info(f"  ↓ {info['label']} ({info['lat']:.4f}, {info['lon']:.4f})...")

        try:
            img = fetch_stitched_image(info["lat"], info["lon"], zoom=zoom, grid_size=grid_size)
            if img is not None:
                img.save(filepath, "JPEG", quality=92)
                saved.append(filepath)
                logger.info(f"    ✓ Saved: {filename} ({img.size[0]}×{img.size[1]})")
            else:
                logger.warning(f"    ✗ Failed to fetch image for {info['label']}")
        except Exception as e:
            logger.error(f"    ✗ Error fetching {info['label']}: {e}")

    # Write a metadata JSON alongside
    import json
    meta_path = os.path.join(output_dir, "metadata.json")
    metadata = {}
    for name, info in INDIAN_HILLS.items():
        filepath = os.path.join(output_dir, f"{name}.jpg")
        metadata[name] = {
            **info,
            "filename": f"{name}.jpg",
            "exists": os.path.isfile(filepath),
            "zoom": zoom,
            "grid_size": grid_size,
        }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    logger.info(f"\nDone! Downloaded {len(saved)}/{len(INDIAN_HILLS)} images.")
    return saved


if __name__ == "__main__":
    fetch_all_indian_hills()
