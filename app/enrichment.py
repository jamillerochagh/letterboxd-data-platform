from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.letterboxd_pipeline.database import (
    get_cached_movies_for,
    upsert_movies,
)
from src.letterboxd_pipeline.tmdb import fetch_movie_metadata


TMDB_FIELDS = [
    "tmdb_id",
    "director",
    "cast_top",
    "genre_primary",
    "genre_secondary",
    "genre_tertiary",
    "country_primary",
    "original_language",
    "runtime_min",
    "vote_average",
    "popularity",
    "tagline",
    "poster_path",
    "overview",
]

MAX_TMDB_WORKERS = 8
PROGRESS_STEP = 5


def empty_metadata() -> dict:
    return {field: None for field in TMDB_FIELDS}


def _normalize_year(value) -> str:
    if value is None or pd.isna(value):
        return ""

    year = str(value).strip()
    if year.endswith(".0"):
        year = year[:-2]

    return year


def _movie_key(title, year) -> tuple[str, str]:
    return (
        str(title or "").strip().lower(),
        _normalize_year(year),
    )


def build_cache_lookup(
    cached_movies: pd.DataFrame,
) -> dict:
    """Create a title/year lookup containing only usable cached rows."""
    if cached_movies is None or cached_movies.empty:
        return {}

    lookup = {}

    for _, row in cached_movies.iterrows():
        title = row.get("title", "")
        year = row.get("release_year", "")
        key = _movie_key(title, year)

        if key[0] and pd.notna(row.get("tmdb_id")):
            lookup[key] = row.to_dict()

    return lookup


def _metadata_from_cached(cached: dict) -> dict:
    return {
        field: cached.get(field)
        for field in TMDB_FIELDS
    }


def _fetch_one_movie(
    key: tuple[str, str],
    title: str,
    year: str,
) -> tuple[tuple[str, str], dict, dict | None, bool]:
    """
    Fetch one movie from TMDB.

    Returns:
        key,
        normalized metadata,
        cache row (or None),
        success flag
    """
    try:
        fetched = fetch_movie_metadata(
            title,
            year,
        ) or {}

        metadata = {
            **empty_metadata(),
            **fetched,
        }

        if not metadata.get("tmdb_id"):
            return key, metadata, None, False

        cache_row = {
            "tmdb_id": metadata.get("tmdb_id"),
            "title": title,
            "release_year": year,
            "director": metadata.get("director"),
            "cast_top": metadata.get("cast_top"),
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

        return key, metadata, cache_row, True

    except Exception as error:
        print(
            f"TMDB ERROR | {title} ({year}) | "
            f"{type(error).__name__}: {error}"
        )
        return key, empty_metadata(), None, False


def enrich_movies(
    watched: pd.DataFrame,
    progress_callback=None,
    max_workers: int = MAX_TMDB_WORKERS,
) -> pd.DataFrame:
    """
    Enrich a Letterboxd watch history efficiently.

    Performance strategy:
    1. Query PostgreSQL only for titles in the current upload.
    2. Resolve cache hits immediately.
    3. Deduplicate cache misses by title/year.
    4. Fetch missing TMDB metadata concurrently.
    5. Batch-upsert new metadata once.
    6. Reassemble rows in the original Letterboxd order.

    PostgreSQL remains an optional cache. Database failures never prevent
    the user's analysis from continuing through TMDB.
    """
    if watched is None or watched.empty:
        return pd.DataFrame(
            columns=list(watched.columns) + TMDB_FIELDS
            if isinstance(watched, pd.DataFrame)
            else TMDB_FIELDS
        )

    watched = watched.copy()
    total = len(watched)

    cached_movies = get_cached_movies_for(
        watched
    )
    cache = build_cache_lookup(
        cached_movies
    )

    row_payloads = []
    missing_movies = {}

    cache_hits = 0
    tmdb_successes = 0
    tmdb_failures = 0

    # First pass is local only: classify cache hits and misses.
    for _, row in watched.iterrows():
        title = str(
            row.get("Name", "")
        ).strip()

        year = _normalize_year(
            row.get("Year", "")
        )

        key = _movie_key(
            title,
            year,
        )

        cached = cache.get(key)

        if cached:
            metadata = _metadata_from_cached(
                cached
            )
            cache_hits += 1
        else:
            metadata = None

            # The same title/year should never trigger duplicate TMDB calls.
            if key not in missing_movies:
                missing_movies[key] = (
                    title,
                    year,
                )

        row_payloads.append(
            (
                row.to_dict(),
                key,
                metadata,
            )
        )

    # Cached rows count as already completed work.
    completed = cache_hits
    last_reported = -1

    def report_progress(force=False):
        nonlocal last_reported

        if not progress_callback or not total:
            return

        if (
            force
            or completed == total
            or completed - last_reported >= PROGRESS_STEP
        ):
            progress_callback(
                min(completed / total, 1.0)
            )
            last_reported = completed

    report_progress(force=True)

    fetched_by_key = {}
    new_movies = []

    # TMDB calls are the slow part, so run only unique misses concurrently.
    if missing_movies:
        worker_count = max(
            1,
            min(
                int(max_workers or 1),
                len(missing_movies),
            ),
        )

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            futures = {
                executor.submit(
                    _fetch_one_movie,
                    key,
                    title,
                    year,
                ): key
                for key, (
                    title,
                    year,
                ) in missing_movies.items()
            }

            for future in as_completed(
                futures
            ):
                key = futures[future]

                try:
                    (
                        _,
                        metadata,
                        cache_row,
                        success,
                    ) = future.result()
                except Exception as error:
                    title, year = missing_movies[
                        key
                    ]
                    print(
                        f"TMDB WORKER ERROR | "
                        f"{title} ({year}) | "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )
                    metadata = empty_metadata()
                    cache_row = None
                    success = False

                fetched_by_key[
                    key
                ] = metadata

                if success:
                    tmdb_successes += 1

                    if cache_row:
                        new_movies.append(
                            cache_row
                        )
                else:
                    tmdb_failures += 1

                completed += 1
                report_progress()

    # Reassemble the original Letterboxd rows in original order.
    results = []

    for row_dict, key, metadata in row_payloads:
        if metadata is None:
            metadata = fetched_by_key.get(
                key,
                empty_metadata(),
            )

        results.append(
            {
                **row_dict,
                **metadata,
            }
        )

    # One database write for the whole upload.
    if new_movies:
        new_df = (
            pd.DataFrame(new_movies)
            .drop_duplicates(
                subset=["tmdb_id"],
                keep="last",
            )
        )

        if not upsert_movies(
            new_df
        ):
            print(
                "DATABASE CACHE | write failed; "
                "analysis continues with in-memory TMDB data"
            )

    completed = total
    report_progress(force=True)

    print(
        "ENRICHMENT SUMMARY | "
        f"movies={total} | "
        f"cache_hits={cache_hits} | "
        f"unique_tmdb_requests={len(missing_movies)} | "
        f"tmdb_successes={tmdb_successes} | "
        f"tmdb_failures={tmdb_failures} | "
        f"workers={min(max_workers, max(1, len(missing_movies)))}"
    )

    enriched = pd.DataFrame(
        results
    )

    for field in TMDB_FIELDS:
        if field not in enriched.columns:
            enriched[field] = pd.NA

    return enriched
