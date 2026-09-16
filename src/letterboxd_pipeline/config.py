import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"


def get_secret(name: str):
    """
    Get configuration from:
    1. Environment variables (.env locally / deployment env)
    2. Streamlit secrets when running on Streamlit Cloud
    """
    value = os.getenv(name)

    if value:
        return value.strip()

    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]

            if value:
                return str(value).strip()

    except Exception:
        pass

    return None


TMDB_API_KEY = get_secret("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3"

ENRICHED_FILE = PROCESSED_DIR / "letterboxd_enriched.csv"
CATALOG_FILE = PROCESSED_DIR / "tmdb_catalog.csv"
SCORED_FILE = PROCESSED_DIR / "letterboxd_enriched_scored.csv"
RECOMMENDATIONS_FILE = PROCESSED_DIR / "letterboxd_recommendations.csv"

PAGES_PER_GENRE = 8
MIN_RATING = 3.5
LIKED_BOOST = 1.5
SCORE_THRESHOLD = 20
MAX_WORKERS = 5

WEIGHTS = {
    "genre_primary": 0.30,
    "genre_secondary": 0.15,
    "director": 0.20,
    "country_primary": 0.10,
    "original_language": 0.10,
    "decade": 0.15,
}


def validate_config():
    if not TMDB_API_KEY:
        raise ValueError(
            "TMDB_API_KEY not found. "
            "Configure TMDB_API_KEY in .env locally or in Streamlit Secrets."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)