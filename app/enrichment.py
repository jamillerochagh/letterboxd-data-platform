import pandas as pd

from src.letterboxd_pipeline.database import get_cached_movies, upsert_movies
from src.letterboxd_pipeline.tmdb import fetch_movie_metadata


TMDB_FIELDS = [
    "tmdb_id", "director", "genre_primary", "genre_secondary",
    "genre_tertiary", "country_primary", "original_language",
    "runtime_min", "vote_average", "popularity", "tagline",
    "poster_path", "overview",
]


def empty_metadata() -> dict:
    return {field: None for field in TMDB_FIELDS}


def build_cache_lookup(cached_movies: pd.DataFrame) -> dict:
    """Create a title/year lookup containing only usable cached rows."""
    if cached_movies.empty:
        return {}

    lookup = {}
    for _, row in cached_movies.iterrows():
        title = str(row.get("title", "")).strip().lower()
        year = str(row.get("release_year", "")).strip()
        if year.endswith(".0"):
            year = year[:-2]

        if title and pd.notna(row.get("tmdb_id")):
            lookup[(title, year)] = row.to_dict()

    return lookup


def enrich_movies(watched: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Enrich Letterboxd movies. PostgreSQL is an optional shared cache:
    if it fails, the analysis continues directly through TMDB.
    """
    cached_movies = get_cached_movies()
    cache = build_cache_lookup(cached_movies)

    results = []
    new_movies = []
    total = len(watched)
    cache_hits = 0
    tmdb_successes = 0
    tmdb_failures = 0

    for position, (_, row) in enumerate(watched.iterrows(), start=1):
        title = str(row.get("Name", "")).strip()
        year = str(row.get("Year", "")).strip()
        if year.endswith(".0"):
            year = year[:-2]

        key = (title.lower(), year)
        cached = cache.get(key)

        if cached:
            metadata = {field: cached.get(field) for field in TMDB_FIELDS}
            cache_hits += 1
        else:
            try:
                fetched = fetch_movie_metadata(title, year) or {}
                metadata = {**empty_metadata(), **fetched}

                if metadata.get("tmdb_id"):
                    tmdb_successes += 1
                    cache_row = {
                        "tmdb_id": metadata.get("tmdb_id"),
                        "title": title,
                        "release_year": year,
                        "director": metadata.get("director"),
                        "genre_primary": metadata.get("genre_primary"),
                        "genre_secondary": metadata.get("genre_secondary"),
                        "genre_tertiary": metadata.get("genre_tertiary"),
                        "country_primary": metadata.get("country_primary"),
                        "original_language": metadata.get("original_language"),
                        "runtime_min": metadata.get("runtime_min"),
                        "vote_average": metadata.get("vote_average"),
                        "popularity": metadata.get("popularity"),
                        "tagline": metadata.get("tagline"),
                        "poster_path": metadata.get("poster_path"),
                        "overview": metadata.get("overview"),
                    }
                    new_movies.append(cache_row)
                    cache[key] = cache_row
                else:
                    tmdb_failures += 1

            except Exception as error:
                tmdb_failures += 1
                print(
                    f"TMDB ERROR | {title} ({year}) | "
                    f"{type(error).__name__}: {error}"
                )
                metadata = empty_metadata()

        results.append({**row.to_dict(), **metadata})

        if progress_callback and total:
            progress_callback(position / total)

    if new_movies:
        new_df = pd.DataFrame(new_movies).drop_duplicates(subset=["tmdb_id"])
        if not upsert_movies(new_df):
            print(
                "DATABASE CACHE | write failed; "
                "analysis continues with in-memory TMDB data"
            )

    print(
        "ENRICHMENT SUMMARY | "
        f"movies={total} | cache_hits={cache_hits} | "
        f"tmdb_successes={tmdb_successes} | tmdb_failures={tmdb_failures}"
    )

    enriched = pd.DataFrame(results)
    for field in TMDB_FIELDS:
        if field not in enriched.columns:
            enriched[field] = pd.NA

    return enriched
