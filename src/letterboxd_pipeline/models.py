from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from .letterboxd import read_csv_file


def create_profile_id() -> str:
    """Create an anonymous identifier for one Letterboxd profile."""

    return str(uuid.uuid4())


def build_movies_table(
    enriched_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create the shared movie metadata table."""

    columns = [
        "tmdb_id",
        "Name",
        "Year",
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

    available = [
        column
        for column in columns
        if column in enriched_df.columns
    ]

    movies = enriched_df[available].copy()

    movies = movies.rename(
        columns={
            "Name": "title",
            "Year": "release_year",
        }
    )

    if "tmdb_id" in movies.columns:
        movies["tmdb_id"] = pd.to_numeric(
            movies["tmdb_id"],
            errors="coerce",
        ).astype("Int64")

        movies = movies.dropna(subset=["tmdb_id"])

        movies = movies.drop_duplicates(
            subset=["tmdb_id"]
        )

    return movies.reset_index(drop=True)


def build_user_movies_table(
    enriched_df: pd.DataFrame,
    profile_id: str,
) -> pd.DataFrame:
    """Create one row per movie associated with a profile."""

    columns = [
        "tmdb_id",
        "Letterboxd URI",
        "rating",
        "liked",
        "has_review",
        "tags",
    ]

    available = [
        column
        for column in columns
        if column in enriched_df.columns
    ]

    user_movies = enriched_df[available].copy()

    user_movies = user_movies.rename(
        columns={
            "Letterboxd URI": "letterboxd_uri",
        }
    )

    user_movies.insert(
        0,
        "profile_id",
        profile_id,
    )

    return user_movies.reset_index(drop=True)


def build_watch_events_table(
    export_dir: str | Path,
    profile_id: str,
) -> pd.DataFrame:
    """
    Preserve every diary entry instead of collapsing
    rewatches into one movie-level record.
    """

    export_dir = Path(export_dir)

    diary_rows = read_csv_file(
        export_dir / "diary.csv"
    )

    diary = pd.DataFrame(diary_rows)

    if diary.empty:
        return pd.DataFrame(
            columns=[
                "watch_event_id",
                "profile_id",
                "letterboxd_uri",
                "title",
                "release_year",
                "watched_date",
                "rewatch",
                "tags",
            ]
        )

    diary = diary.rename(
        columns={
            "Letterboxd URI": "letterboxd_uri",
            "Name": "title",
            "Year": "release_year",
            "Watched Date": "watched_date",
            "Rewatch": "rewatch",
            "Tags": "tags",
        }
    )

    diary.insert(
        0,
        "profile_id",
        profile_id,
    )

    diary.insert(
        0,
        "watch_event_id",
        [
            str(uuid.uuid4())
            for _ in range(len(diary))
        ],
    )

    return diary.reset_index(drop=True)