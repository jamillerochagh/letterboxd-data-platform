import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


load_dotenv()


def get_database_url() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "letterboxd")
    user = os.getenv("POSTGRES_USER", "letterboxd")
    password = os.getenv("POSTGRES_PASSWORD", "letterboxd")

    return (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )


def get_engine():
    """
    Create a PostgreSQL connection.

    Production:
        Uses DATABASE_URL (Supabase).

    Local development:
        Uses the individual PostgreSQL variables from .env.
    """

    database_url = os.getenv("DATABASE_URL")

    # Production / hosted database
    if database_url:
        return create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=300,
        )

    # Local Docker PostgreSQL
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

    local_url = (
        f"postgresql+psycopg2://"
        f"{user}:{password}"
        f"@{host}:{port}/{database}"
    )

    return create_engine(
        local_url,
        pool_pre_ping=True,
    )


def test_connection() -> bool:
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))

    return result.scalar() == 1


def load_dataframe(
    df: pd.DataFrame,
    table_name: str,
    schema: str = "raw",
) -> None:
    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(
            text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
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
        f"Loaded {len(df)} rows into "
        f"{schema}.{table_name}"
    )
def get_cached_movies() -> pd.DataFrame:
    """Load the shared TMDB movie cache from PostgreSQL."""

    engine = get_engine()

    query = """
        SELECT *
        FROM raw.movies
    """

    try:
        return pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame()


def upsert_movies(df: pd.DataFrame) -> None:
    """Add new movies to the shared PostgreSQL movie cache."""

    if df.empty:
        return

    engine = get_engine()

    existing = get_cached_movies()

    if not existing.empty and "tmdb_id" in existing.columns:
        existing_ids = set(
            pd.to_numeric(
                existing["tmdb_id"],
                errors="coerce",
            ).dropna().astype(int)
        )

        new_ids = pd.to_numeric(
            df["tmdb_id"],
            errors="coerce",
        )

        df = df[
            ~new_ids.isin(existing_ids)
        ].copy()

    if df.empty:
        return

    df.to_sql(
        name="movies",
        con=engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )

    print(f"Cached {len(df)} new movies.")   