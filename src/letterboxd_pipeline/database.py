import os
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()


def get_secret(name: str):
    """
    Load configuration from environment variables first,
    then Streamlit Secrets when running on Streamlit Cloud.
    """
    value = os.getenv(name)

    if value:
        return str(value).strip()

    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]

            if value:
                return str(value).strip()

    except Exception:
        pass

    return None


def get_database_url() -> str:
    """
    Return the production DATABASE_URL when available.
    Otherwise build the local PostgreSQL URL.
    """

    database_url = get_secret("DATABASE_URL")

    if database_url:
        return database_url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "letterboxd")
    user = os.getenv("POSTGRES_USER", "letterboxd")
    password = os.getenv("POSTGRES_PASSWORD", "letterboxd")

    return (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )


def log_database_config(database_url: str) -> None:
    """
    Print safe connection diagnostics without exposing the password.
    """

    try:
        parsed = urlparse(database_url)

        print(
            "DATABASE CONFIG | "
            f"host={parsed.hostname} | "
            f"port={parsed.port} | "
            f"user={parsed.username} | "
            f"database={parsed.path.lstrip('/')}"
        )

    except Exception as error:
        print(
            "DATABASE CONFIG ERROR | "
            f"{type(error).__name__}: {error}"
        )


def get_engine():
    """
    Create a PostgreSQL connection.

    Production:
        Uses DATABASE_URL from environment variables
        or Streamlit Secrets.

    Local development:
        Uses individual PostgreSQL variables from .env.
    """

    database_url = get_database_url()

    log_database_config(database_url)

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
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
    """
    Load the shared TMDB movie cache from PostgreSQL.
    """

    engine = get_engine()

    query = """
        SELECT *
        FROM raw.movies
    """

    try:
        movies = pd.read_sql(query, engine)

        print(
            "DATABASE CACHE | "
            f"loaded={len(movies)} movies"
        )

        return movies

    except Exception as error:
        print(
            "DATABASE CACHE ERROR | "
            f"{type(error).__name__}: {error}"
        )

        return pd.DataFrame()


def upsert_movies(df: pd.DataFrame) -> None:
    """
    Add new movies to the shared PostgreSQL movie cache.
    """

    if df.empty:
        return

    engine = get_engine()

    existing = get_cached_movies()

    if not existing.empty and "tmdb_id" in existing.columns:
        existing_ids = set(
            pd.to_numeric(
                existing["tmdb_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
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

    print(
        "DATABASE CACHE | "
        f"cached={len(df)} new movies"
    )