import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import ENRICHED_FILE
from .letterboxd import load_letterboxd_export
from .tmdb import fetch_movie_metadata


def load_enrichment_cache(cache_file: Path = ENRICHED_FILE) -> dict:
    """
    Load previously enriched movies.

    Letterboxd URI is used as the unique key so movies already processed
    do not require another TMDB request.
    """

    if not cache_file.exists():
        return {}

    try:
        cached = pd.read_csv(
        cache_file,
        dtype=str,
        keep_default_na=False,
    )

        if "Letterboxd URI" not in cached.columns:
            return {}

        return {
            row["Letterboxd URI"]: row.to_dict()
            for _, row in cached.iterrows()
        }

    except Exception:
        return {}


def build_internal_metadata(
    uri: str,
    ratings: dict,
    diary: dict,
    reviews: set,
    likes: set,
) -> dict:
    """Build metadata coming directly from the Letterboxd export."""

    diary_info = diary.get(uri, {})

    return {
        "rating": ratings.get(uri, ""),
        "watched_date": diary_info.get("watched_date", ""),
        "rewatch": diary_info.get("rewatch", ""),
        "tags": diary_info.get("tags", ""),
        "has_review": "Yes" if uri in reviews else "No",
        "liked": "Yes" if uri in likes else "No",
    }


def get_cached_tmdb_metadata(cached_row: dict) -> dict:
    """Extract only TMDB fields from an existing enriched record."""

    fields = [
        "tmdb_id",
        "director",
        "genre_primary",
        "genre_secondary",
        "genre_tertiary",
        "country_primary",
        "original_language",
        "runtime_min",
        "vote_average",
        "popularity",
        "tagline",
        "overview",
    ]

    return {
        field: cached_row.get(field, "")
        for field in fields
    }


def enrich_letterboxd_export(
    export_dir: str | Path,
    output_file: Path = ENRICHED_FILE,
) -> pd.DataFrame:
    """
    Enrich a Letterboxd export with TMDB metadata.

    Existing TMDB metadata is reused whenever possible, making subsequent
    runs incremental.
    """

    export = load_letterboxd_export(export_dir)

    watched = export["watched"]
    ratings = export["ratings"]
    diary = export["diary"]
    reviews = export["reviews"]
    likes = export["likes"]

    cache = load_enrichment_cache(output_file)

    enriched_rows = []

    cache_hits = 0
    api_requests = 0
    failed_requests = 0

    print(f"\nFound {len(watched)} films in Letterboxd export.")
    print(f"Found {len(cache)} films in enrichment cache.\n")

    for row in tqdm(watched, desc="Enriching movies", unit="film"):

        title = row.get("Name", "").strip()
        year = row.get("Year", "").strip()
        uri = row.get("Letterboxd URI", "")

        internal_metadata = build_internal_metadata(
            uri=uri,
            ratings=ratings,
            diary=diary,
            reviews=reviews,
            likes=likes,
        )

        # Reuse TMDB metadata if the movie was processed before.
        cached_row = cache.get(uri)
        if (
            cached_row
            and str(
                cached_row.get("tmdb_id", "")
            ).strip()
        ):
            tmdb_metadata = get_cached_tmdb_metadata(cache[uri])
            cache_hits += 1

        else:
            try:
                tmdb_metadata = fetch_movie_metadata(title, year)
                api_requests += 1

            except Exception as error:
                failed_requests += 1

                print(
                    f"\nTMDB request failed for "
                    f"{title} ({year}): {error}"
                )

                tmdb_metadata = {
                    "tmdb_id": "",
                    "director": "",
                    "genre_primary": "",
                    "genre_secondary": "",
                    "genre_tertiary": "",
                    "country_primary": "",
                    "original_language": "",
                    "runtime_min": "",
                    "vote_average": "",
                    "popularity": "",
                    "tagline": "",
                    "overview": "",
                }

            time.sleep(0.20)

        enriched_rows.append(
            {
                **row,
                **internal_metadata,
                **tmdb_metadata,
            }
        )

    enriched_df = pd.DataFrame(enriched_rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    enriched_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print("\nEnrichment complete.")
    print(f"Cache hits: {cache_hits}")
    print(f"TMDB API requests: {api_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Output: {output_file}")

    return enriched_df