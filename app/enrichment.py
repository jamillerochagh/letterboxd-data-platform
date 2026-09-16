import pandas as pd

from src.letterboxd_pipeline.database import (
    get_cached_movies,
    upsert_movies,
)
from src.letterboxd_pipeline.tmdb import (
    fetch_movie_metadata,
)


def build_cache_lookup(
    cached_movies: pd.DataFrame,
) -> dict:
    """Create title/year lookup for cached movie metadata."""

    if cached_movies.empty:
        return {}

    lookup = {}

    for _, row in cached_movies.iterrows():
        title = str(row.get("title", "")).strip().lower()
        year = str(row.get("release_year", "")).strip()

        if year.endswith(".0"):
            year = year[:-2]

        if title:
            lookup[(title, year)] = row.to_dict()

    return lookup


def enrich_movies(
    watched: pd.DataFrame,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Enrich uploaded Letterboxd movies.

    PostgreSQL is checked first. TMDB is only called
    when metadata is not already cached.
    """

    cached_movies = get_cached_movies()
    cache = build_cache_lookup(cached_movies)

    results = []
    new_movies = []

    total = len(watched)

    for position, (_, row) in enumerate(
        watched.iterrows(),
        start=1,
    ):
        title = str(row.get("Name", "")).strip()
        year = str(row.get("Year", "")).strip()

        if year.endswith(".0"):
            year = year[:-2]

        key = (
            title.lower(),
            year,
        )

        cached = cache.get(key)

        if cached:
            metadata = cached

        else:
            try:
                metadata = fetch_movie_metadata(
                    title,
                    year,
                )
            except Exception as error:
                print(
                    f"TMDB ERROR | {title} ({year}) | "
                    f"{type(error).__name__}: {error}"
                )
                metadata = {}

            if metadata.get("tmdb_id"):
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

        results.append(
            {
                **row.to_dict(),
                **metadata,
            }
        )

        if progress_callback:
            progress_callback(
                position / total
            )

    if new_movies:
        new_df = (
            pd.DataFrame(new_movies)
            .drop_duplicates(subset=["tmdb_id"])
        )

        upsert_movies(new_df)

    return pd.DataFrame(results)