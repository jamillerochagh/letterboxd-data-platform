import os
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import make_url

load_dotenv()


MOVIE_CACHE_COLUMNS = [
    "tmdb_id",
    "title",
    "release_year",
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
    "overview",
    "poster_path",
]


def get_secret(name: str):
    """Prefer Streamlit Secrets in production, then environment variables."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]

            if value:
                return str(value).strip()

    except Exception:
        pass

    value = os.getenv(name)

    return (
        str(value).strip()
        if value
        else None
    )


def get_database_url() -> str:
    database_url = get_secret(
        "DATABASE_URL"
    )

    if database_url:
        return database_url

    host = os.getenv(
        "POSTGRES_HOST",
        "localhost",
    )
    port = os.getenv(
        "POSTGRES_PORT",
        "5432",
    )
    database = os.getenv(
        "POSTGRES_DB",
        "letterboxd",
    )
    user = os.getenv(
        "POSTGRES_USER",
        "letterboxd",
    )
    password = os.getenv(
        "POSTGRES_PASSWORD",
        "letterboxd",
    )

    return (
        f"postgresql+psycopg2://"
        f"{user}:{password}@"
        f"{host}:{port}/{database}"
    )


def log_database_config(
    database_url: str,
) -> None:
    """Log connection identity without printing the password."""
    try:
        parsed = make_url(
            database_url
        )

        print(
            "DATABASE CONFIG | "
            f"host={parsed.host} | "
            f"port={parsed.port} | "
            f"user={parsed.username} | "
            f"database={parsed.database}"
        )

    except Exception as error:
        print(
            "DATABASE CONFIG ERROR | "
            f"{type(error).__name__}: "
            f"{error}"
        )


@lru_cache(maxsize=1)
def get_engine():
    """
    Reuse one SQLAlchemy engine per app process.

    Creating a new engine for every cache read/write adds unnecessary
    connection-pool setup overhead.
    """
    database_url = get_database_url()

    log_database_config(
        database_url
    )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=5,
        connect_args={
            "connect_timeout": 10,
        },
    )


def test_connection() -> bool:
    try:
        engine = get_engine()

        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT 1")
            )

        return result.scalar() == 1

    except Exception as error:
        print(
            "DATABASE CONNECTION ERROR | "
            f"{type(error).__name__}: "
            f"{error}"
        )
        return False


def load_dataframe(
    df: pd.DataFrame,
    table_name: str,
    schema: str = "raw",
) -> bool:
    if df.empty:
        return True

    try:
        engine = get_engine()

        with engine.begin() as connection:
            connection.execute(
                text(
                    f'CREATE SCHEMA IF NOT EXISTS "{schema}"'
                )
            )

        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=500,
        )

        print(
            f"Loaded {len(df)} rows "
            f"into {schema}.{table_name}"
        )
        return True

    except Exception as error:
        print(
            "DATABASE LOAD ERROR | "
            f"{type(error).__name__}: "
            f"{error}"
        )
        return False


def get_cached_movies() -> pd.DataFrame:
    """
    Read the complete shared cache.

    Kept for maintenance/backfill scripts. User-facing enrichment should
    prefer get_cached_movies_for() so large caches are not downloaded for
    every upload.
    """
    try:
        engine = get_engine()

        movies = pd.read_sql(
            "SELECT * FROM raw.movies",
            engine,
        )

        print(
            "DATABASE CACHE | "
            f"loaded={len(movies)} movies"
        )

        return movies

    except Exception as error:
        print(
            "DATABASE CACHE ERROR | "
            f"{type(error).__name__}: "
            f"{error}"
        )
        return pd.DataFrame()


def get_cached_movies_for(
    watched: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load only cache rows relevant to the current Letterboxd upload.

    Matching is narrowed by normalized title in PostgreSQL and finalized
    by title/year in app/enrichment.py. Chunking avoids oversized IN
    clauses for large Letterboxd histories.
    """
    if (
        watched is None
        or watched.empty
        or "Name" not in watched.columns
    ):
        return pd.DataFrame(
            columns=MOVIE_CACHE_COLUMNS
        )

    titles = (
        watched["Name"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )

    titles = sorted(
        {
            title
            for title in titles
            if title
        }
    )

    if not titles:
        return pd.DataFrame(
            columns=MOVIE_CACHE_COLUMNS
        )

    try:
        engine = get_engine()

        frames = []
        chunk_size = 500

        statement = text(
            """
            SELECT
                tmdb_id,
                title,
                release_year,
                director,
                cast_top,
                genre_primary,
                genre_secondary,
                genre_tertiary,
                country_primary,
                original_language,
                runtime_min,
                vote_average,
                popularity,
                tagline,
                overview,
                poster_path
            FROM raw.movies
            WHERE LOWER(title) IN :titles
            """
        ).bindparams(
            bindparam(
                "titles",
                expanding=True,
            )
        )

        for start in range(
            0,
            len(titles),
            chunk_size,
        ):
            chunk = titles[
                start:start + chunk_size
            ]

            frame = pd.read_sql(
                statement,
                engine,
                params={
                    "titles": chunk,
                },
            )

            if not frame.empty:
                frames.append(
                    frame
                )

        if not frames:
            print(
                "DATABASE CACHE | "
                "matched=0 movies"
            )
            return pd.DataFrame(
                columns=MOVIE_CACHE_COLUMNS
            )

        movies = (
            pd.concat(
                frames,
                ignore_index=True,
            )
            .drop_duplicates(
                subset=["tmdb_id"],
                keep="last",
            )
        )

        print(
            "DATABASE CACHE | "
            f"matched={len(movies)} movies | "
            f"requested_titles={len(titles)}"
        )

        return movies

    except Exception as error:
        print(
            "DATABASE CACHE FILTER ERROR | "
            f"{type(error).__name__}: "
            f"{error}"
        )

        # Database cache is optional. Do not fall back to SELECT * because
        # that recreates the performance problem this function avoids.
        return pd.DataFrame(
            columns=MOVIE_CACHE_COLUMNS
        )


def ensure_movies_cast_column() -> bool:
    """Add cast_top to legacy caches without breaking older deployments."""
    try:
        engine = get_engine()

        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE raw.movies "
                    "ADD COLUMN IF NOT EXISTS cast_top TEXT"
                )
            )

        return True

    except Exception as error:
        print(
            "DATABASE MIGRATION ERROR | "
            f"{type(error).__name__}: "
            f"{error}"
        )
        return False


def upsert_movies(
    df: pd.DataFrame,
) -> bool:
    """
    Batch insert/update TMDB metadata.

    PostgreSQL cache failure remains non-fatal.
    """
    if (
        df.empty
        or "tmdb_id" not in df.columns
    ):
        return True

    try:
        engine = get_engine()

        # Safe for old databases; PostgreSQL IF NOT EXISTS makes this
        # idempotent.
        ensure_movies_cast_column()

        clean_df = df.copy()

        clean_df["tmdb_id"] = pd.to_numeric(
            clean_df["tmdb_id"],
            errors="coerce",
        )

        clean_df = clean_df.dropna(
            subset=["tmdb_id"]
        )

        clean_df["tmdb_id"] = (
            clean_df["tmdb_id"]
            .astype("int64")
        )

        clean_df = (
            clean_df
            .drop_duplicates(
                subset=["tmdb_id"],
                keep="last",
            )
        )

        if clean_df.empty:
            return True

        columns = [
            column
            for column in clean_df.columns
            if column in set(
                MOVIE_CACHE_COLUMNS
            )
        ]

        clean_df = clean_df[
            columns
        ].copy()

        records = (
            clean_df
            .where(
                pd.notna(clean_df),
                None,
            )
            .to_dict(
                orient="records"
            )
        )

        if not records:
            return True

        quoted_columns = ", ".join(
            f'"{column}"'
            for column in columns
        )

        value_columns = ", ".join(
            f":{column}"
            for column in columns
        )

        update_columns = [
            column
            for column in columns
            if column != "tmdb_id"
        ]

        if update_columns:
            update_clause = ", ".join(
                f'"{column}" = '
                f'EXCLUDED."{column}"'
                for column in update_columns
            )

            conflict_clause = (
                "DO UPDATE SET "
                + update_clause
            )
        else:
            conflict_clause = "DO NOTHING"

        statement = text(
            f"""
            INSERT INTO raw.movies (
                {quoted_columns}
            )
            VALUES (
                {value_columns}
            )
            ON CONFLICT (tmdb_id)
            WHERE tmdb_id IS NOT NULL
            {conflict_clause}
            """
        )

        with engine.begin() as connection:
            connection.execute(
                statement,
                records,
            )

        print(
            "DATABASE CACHE | "
            f"upserted={len(records)} movies"
        )

        return True

    except Exception as error:
        print(
            "DATABASE CACHE WRITE ERROR | "
            f"{type(error).__name__}: "
            f"{error}"
        )
        return False
