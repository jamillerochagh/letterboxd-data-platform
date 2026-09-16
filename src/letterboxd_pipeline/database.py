import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

load_dotenv()


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
    return str(value).strip() if value else None


def get_database_url() -> str:
    database_url = get_secret("DATABASE_URL")
    if database_url:
        return database_url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "letterboxd")
    user = os.getenv("POSTGRES_USER", "letterboxd")
    password = os.getenv("POSTGRES_PASSWORD", "letterboxd")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def log_database_config(database_url: str) -> None:
    """Log connection identity without printing the password."""
    try:
        parsed = make_url(database_url)
        print(
            "DATABASE CONFIG | "
            f"host={parsed.host} | port={parsed.port} | "
            f"user={parsed.username} | database={parsed.database}"
        )
    except Exception as error:
        print(f"DATABASE CONFIG ERROR | {type(error).__name__}: {error}")


def get_engine():
    database_url = get_database_url()
    log_database_config(database_url)
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"connect_timeout": 10},
    )


def test_connection() -> bool:
    try:
        engine = get_engine()
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
        return result.scalar() == 1
    except Exception as error:
        print(f"DATABASE CONNECTION ERROR | {type(error).__name__}: {error}")
        return False


def load_dataframe(df: pd.DataFrame, table_name: str, schema: str = "raw") -> bool:
    if df.empty:
        return True
    try:
        engine = get_engine()
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        df.to_sql(
            name=table_name, con=engine, schema=schema, if_exists="replace",
            index=False, method="multi", chunksize=500,
        )
        print(f"Loaded {len(df)} rows into {schema}.{table_name}")
        return True
    except Exception as error:
        print(f"DATABASE LOAD ERROR | {type(error).__name__}: {error}")
        return False


def get_cached_movies() -> pd.DataFrame:
    """Read the shared cache; database failure must not break user analysis."""
    try:
        engine = get_engine()
        movies = pd.read_sql("SELECT * FROM raw.movies", engine)
        print(f"DATABASE CACHE | loaded={len(movies)} movies")
        return movies
    except Exception as error:
        print(f"DATABASE CACHE ERROR | {type(error).__name__}: {error}")
        return pd.DataFrame()


def upsert_movies(df: pd.DataFrame) -> bool:
    """Write to shared cache; failure is non-fatal because TMDB data stays in memory."""
    if df.empty or "tmdb_id" not in df.columns:
        return True

    try:
        engine = get_engine()
        existing = get_cached_movies()

        clean_df = df.copy()
        clean_df["tmdb_id"] = pd.to_numeric(clean_df["tmdb_id"], errors="coerce")
        clean_df = clean_df.dropna(subset=["tmdb_id"])
        clean_df["tmdb_id"] = clean_df["tmdb_id"].astype("int64")
        clean_df = clean_df.drop_duplicates(subset=["tmdb_id"])

        if not existing.empty and "tmdb_id" in existing.columns:
            existing_ids = set(
                pd.to_numeric(existing["tmdb_id"], errors="coerce")
                .dropna().astype("int64")
            )
            clean_df = clean_df[~clean_df["tmdb_id"].isin(existing_ids)].copy()

        if clean_df.empty:
            return True

        clean_df.to_sql(
            name="movies", con=engine, schema="raw", if_exists="append",
            index=False, method="multi", chunksize=500,
        )
        print(f"DATABASE CACHE | cached={len(clean_df)} new movies")
        return True

    except Exception as error:
        print(f"DATABASE CACHE WRITE ERROR | {type(error).__name__}: {error}")
        return False
