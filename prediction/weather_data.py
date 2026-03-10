"""
Live Weather & Soil Data Module
==================================
Fetches real-time weather and soil moisture data from the Open-Meteo API
(free, no API key required) for landslide risk prediction.

Data includes:
  - Temperature, humidity, precipitation
  - Soil moisture at multiple depths
  - Wind speed, cloud cover
  - 7-day rainfall history

API docs: https://open-meteo.com/en/docs
"""

import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ================================================================
# INDIAN MONITORING STATIONS
# ================================================================

MONITORING_STATIONS = {
    # --- Western Ghats ---
    "Kodaikanal, Tamil Nadu":       {"lat": 10.2381, "lon": 77.4892, "elevation": 2133, "region": "Western Ghats"},
    "Ooty, Tamil Nadu":             {"lat": 11.4102, "lon": 76.6950, "elevation": 2240, "region": "Nilgiris"},
    "Munnar, Kerala":               {"lat": 10.0889, "lon": 77.0595, "elevation": 1532, "region": "Western Ghats"},
    "Wayanad, Kerala":              {"lat": 11.6854, "lon": 76.1320, "elevation": 936,  "region": "Western Ghats"},
    "Idukki, Kerala":               {"lat": 9.8494,  "lon": 76.9710, "elevation": 1200, "region": "Western Ghats"},
    "Coorg, Karnataka":             {"lat": 12.3375, "lon": 75.8069, "elevation": 1170, "region": "Western Ghats"},
    "Chikmagalur, Karnataka":       {"lat": 13.3161, "lon": 75.7720, "elevation": 1090, "region": "Western Ghats"},
    "Mahabaleshwar, Maharashtra":   {"lat": 17.9307, "lon": 73.6477, "elevation": 1353, "region": "Western Ghats"},
    # --- Himalayas ---
    "Kedarnath, Uttarakhand":       {"lat": 30.7352, "lon": 79.0669, "elevation": 3583, "region": "Himalayas"},
    "Joshimath, Uttarakhand":       {"lat": 30.5550, "lon": 79.5650, "elevation": 1890, "region": "Himalayas"},
    "Shimla, Himachal Pradesh":     {"lat": 31.1048, "lon": 77.1734, "elevation": 2276, "region": "Himalayas"},
    "Manali, Himachal Pradesh":     {"lat": 32.2396, "lon": 77.1887, "elevation": 2050, "region": "Himalayas"},
    "Dharamshala, Himachal Pradesh":{"lat": 32.2190, "lon": 76.3234, "elevation": 1457, "region": "Himalayas"},
    "Gangtok, Sikkim":              {"lat": 27.3389, "lon": 88.6065, "elevation": 1650, "region": "Himalayas"},
    "Darjeeling, West Bengal":      {"lat": 27.0360, "lon": 88.2627, "elevation": 2042, "region": "Himalayas"},
    # --- Northeast India ---
    "Shillong, Meghalaya":          {"lat": 25.5788, "lon": 91.8933, "elevation": 1496, "region": "Northeast"},
    "Cherrapunji, Meghalaya":       {"lat": 25.2700, "lon": 91.7200, "elevation": 1484, "region": "Northeast"},
    "Aizawl, Mizoram":              {"lat": 23.7271, "lon": 92.7176, "elevation": 1132, "region": "Northeast"},
    "Itanagar, Arunachal Pradesh":  {"lat": 27.0844, "lon": 93.6053, "elevation": 320,  "region": "Northeast"},
}


# ================================================================
# WEATHER DATA FETCHER
# ================================================================

class WeatherDataFetcher:
    """
    Fetches live weather and soil data from Open-Meteo API.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    CURRENT_PARAMS = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "weather_code",
        "cloud_cover",
        "wind_speed_10m",
        "soil_moisture_0_to_1cm",
        "soil_moisture_1_to_3cm",
        "soil_moisture_3_to_9cm",
        "soil_moisture_9_to_27cm",
        "soil_temperature_0cm",
    ]

    DAILY_PARAMS = [
        "precipitation_sum",
        "rain_sum",
        "precipitation_hours",
        "temperature_2m_max",
        "temperature_2m_min",
    ]

    # WMO Weather codes → descriptions
    WEATHER_CODES = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + heavy hail",
    }

    _cache = {}
    _cache_ttl = 300  # 5 min

    @classmethod
    def fetch(cls, lat, lon, station_name="Unknown"):
        """
        Fetch current weather + 7-day history for given coordinates.

        Returns:
            dict with keys: current, daily, station, fetched_at, source
        """
        cache_key = f"{lat:.2f},{lon:.2f}"
        now = time.time()
        if cache_key in cls._cache and (now - cls._cache[cache_key]["_ts"]) < cls._cache_ttl:
            logger.debug(f"Using cached weather for {station_name}")
            return cls._cache[cache_key]

        if not HAS_REQUESTS:
            return cls._generate_fallback(station_name, lat, lon)

        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": ",".join(cls.CURRENT_PARAMS),
                "daily": ",".join(cls.DAILY_PARAMS),
                "timezone": "Asia/Kolkata",
                "past_days": 7,
                "forecast_days": 1,
            }

            resp = requests.get(cls.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()

            data = cls._parse_response(raw, station_name, lat, lon)
            data["_ts"] = now
            data["source"] = "live"
            cls._cache[cache_key] = data

            logger.info(f"Fetched live weather for {station_name}")
            return data

        except Exception as e:
            logger.warning(f"Weather API failed for {station_name}: {e}")
            return cls._generate_fallback(station_name, lat, lon)

    @classmethod
    def _parse_response(cls, raw, station_name, lat, lon):
        """Parse Open-Meteo response into clean structure."""
        current = raw.get("current", {})
        daily = raw.get("daily", {})

        # Current conditions
        weather_code = current.get("weather_code", 0)
        weather_desc = cls.WEATHER_CODES.get(weather_code, "Unknown")

        # Soil moisture — average across depths
        sm_0_1 = current.get("soil_moisture_0_to_1cm", 0) or 0
        sm_1_3 = current.get("soil_moisture_1_to_3cm", 0) or 0
        sm_3_9 = current.get("soil_moisture_3_to_9cm", 0) or 0
        sm_9_27 = current.get("soil_moisture_9_to_27cm", 0) or 0
        avg_soil_moisture = (sm_0_1 + sm_1_3 + sm_3_9 + sm_9_27) / 4

        # 7-day rainfall total
        precip_daily = daily.get("precipitation_sum", [])
        rainfall_7d = sum(p for p in precip_daily if p is not None)
        rain_hours = daily.get("precipitation_hours", [])
        rain_hours_7d = sum(h for h in rain_hours if h is not None)

        return {
            "station": station_name,
            "lat": lat,
            "lon": lon,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "current": {
                "temperature": current.get("temperature_2m", 0),
                "humidity": current.get("relative_humidity_2m", 0),
                "precipitation": current.get("precipitation", 0),
                "rain": current.get("rain", 0),
                "wind_speed": current.get("wind_speed_10m", 0),
                "cloud_cover": current.get("cloud_cover", 0),
                "weather_code": weather_code,
                "weather_description": weather_desc,
                "soil_moisture_surface": round(sm_0_1 * 100, 1),  # %
                "soil_moisture_shallow": round(sm_1_3 * 100, 1),
                "soil_moisture_mid": round(sm_3_9 * 100, 1),
                "soil_moisture_deep": round(sm_9_27 * 100, 1),
                "soil_moisture_avg": round(avg_soil_moisture * 100, 1),
                "soil_temperature": current.get("soil_temperature_0cm", 0),
            },
            "history": {
                "rainfall_7d_mm": round(rainfall_7d, 1),
                "rain_hours_7d": round(rain_hours_7d, 1),
                "daily_precip": precip_daily,
                "daily_temp_max": daily.get("temperature_2m_max", []),
                "daily_temp_min": daily.get("temperature_2m_min", []),
                "dates": daily.get("time", []),
            },
        }

    @staticmethod
    def _generate_fallback(station_name, lat, lon):
        """Generate synthetic weather data when API is unavailable."""
        import random
        random.seed(int(lat * 100 + lon * 10))

        return {
            "station": station_name,
            "lat": lat,
            "lon": lon,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "source": "simulated",
            "current": {
                "temperature": round(random.uniform(12, 28), 1),
                "humidity": round(random.uniform(60, 95), 0),
                "precipitation": round(random.uniform(0, 8), 1),
                "rain": round(random.uniform(0, 5), 1),
                "wind_speed": round(random.uniform(2, 25), 1),
                "cloud_cover": round(random.uniform(20, 100), 0),
                "weather_code": random.choice([0, 3, 61, 63, 80, 95]),
                "weather_description": random.choice(["Clear sky", "Moderate rain", "Overcast", "Slight rain"]),
                "soil_moisture_surface": round(random.uniform(10, 60), 1),
                "soil_moisture_shallow": round(random.uniform(15, 55), 1),
                "soil_moisture_mid": round(random.uniform(20, 50), 1),
                "soil_moisture_deep": round(random.uniform(25, 45), 1),
                "soil_moisture_avg": round(random.uniform(18, 50), 1),
                "soil_temperature": round(random.uniform(10, 25), 1),
            },
            "history": {
                "rainfall_7d_mm": round(random.uniform(5, 200), 1),
                "rain_hours_7d": round(random.uniform(2, 60), 1),
                "daily_precip": [round(random.uniform(0, 30), 1) for _ in range(8)],
                "daily_temp_max": [round(random.uniform(18, 32), 1) for _ in range(8)],
                "daily_temp_min": [round(random.uniform(8, 20), 1) for _ in range(8)],
                "dates": [(datetime.now() - timedelta(days=7-i)).strftime("%Y-%m-%d") for i in range(8)],
            },
        }


# ================================================================
# RISK CALCULATOR (WEATHER-BASED)
# ================================================================

class WeatherRiskCalculator:
    """
    Calculates landslide risk based on weather and soil conditions.
    """

    @staticmethod
    def calculate(weather_data):
        """
        Calculate weather-based risk score.

        Returns:
            dict with overall_score, factors breakdown, alert_level
        """
        current = weather_data.get("current", {})
        history = weather_data.get("history", {})

        # --- Factor 1: Current rainfall intensity (0-100) ---
        rain_now = current.get("precipitation", 0) + current.get("rain", 0)
        if rain_now > 20:
            rainfall_score = 100
        elif rain_now > 10:
            rainfall_score = 80
        elif rain_now > 5:
            rainfall_score = 60
        elif rain_now > 1:
            rainfall_score = 35
        else:
            rainfall_score = 10

        # --- Factor 2: 7-day cumulative rainfall (0-100) ---
        rain_7d = history.get("rainfall_7d_mm", 0)
        if rain_7d > 200:
            cumulative_score = 100
        elif rain_7d > 100:
            cumulative_score = 80
        elif rain_7d > 50:
            cumulative_score = 55
        elif rain_7d > 20:
            cumulative_score = 30
        else:
            cumulative_score = 10

        # --- Factor 3: Soil moisture saturation (0-100) ---
        soil_avg = current.get("soil_moisture_avg", 0)
        if soil_avg > 45:
            soil_score = 100
        elif soil_avg > 35:
            soil_score = 75
        elif soil_avg > 25:
            soil_score = 50
        elif soil_avg > 15:
            soil_score = 25
        else:
            soil_score = 10

        # --- Factor 4: Humidity (0-100) ---
        humidity = current.get("humidity", 0)
        if humidity > 90:
            humidity_score = 85
        elif humidity > 80:
            humidity_score = 60
        elif humidity > 70:
            humidity_score = 40
        else:
            humidity_score = 15

        # --- Weighted combination ---
        weights = {
            "rainfall_intensity": 0.30,
            "cumulative_rainfall": 0.30,
            "soil_saturation": 0.25,
            "humidity": 0.15,
        }

        overall = (
            rainfall_score * weights["rainfall_intensity"]
            + cumulative_score * weights["cumulative_rainfall"]
            + soil_score * weights["soil_saturation"]
            + humidity_score * weights["humidity"]
        )

        # --- Alert level ---
        if overall >= 75:
            alert = "DANGER"
        elif overall >= 50:
            alert = "WARNING"
        elif overall >= 30:
            alert = "WATCH"
        else:
            alert = "SAFE"

        return {
            "overall_score": round(overall, 1),
            "alert_level": alert,
            "factors": {
                "rainfall_intensity": {"score": rainfall_score, "weight": weights["rainfall_intensity"],
                                       "value": rain_now, "unit": "mm/h"},
                "cumulative_rainfall": {"score": cumulative_score, "weight": weights["cumulative_rainfall"],
                                        "value": rain_7d, "unit": "mm (7d)"},
                "soil_saturation": {"score": soil_score, "weight": weights["soil_saturation"],
                                    "value": soil_avg, "unit": "%"},
                "humidity": {"score": humidity_score, "weight": weights["humidity"],
                             "value": humidity, "unit": "%"},
            },
        }
