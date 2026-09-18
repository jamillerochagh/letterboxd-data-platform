"""
Populate raw.movies.cast_top without slowing down the Streamlit app.

Run from the project root:
    source .venv/bin/activate
    PYTHONPATH=. python scripts/backfill_cast.py
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import create_engine, text

from src.letterboxd_pipeline.database import get_database_url
from src.letterboxd_pipeline.tmdb import get_movie_details


MAX_WORKERS = 6
BATCH_LIMIT = 5000
CAST_SIZE = 5


def load_missing_movies(engine):
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT tmdb_id, title, release_year
                FROM raw.movies
                WHERE tmdb_id IS NOT NULL
                  AND (cast_top IS NULL OR btrim(cast_top) = '')
                ORDER BY tmdb_id
                LIMIT :limit
                """
            ),
            {"limit": BATCH_LIMIT},
        )
        return [dict(row._mapping) for row in rows]


def extract_cast(details):
    credits = (details or {}).get("credits", {})
    cast = credits.get("cast", [])

    names = [
        person.get("name", "").strip()
        for person in cast[:CAST_SIZE]
        if person.get("name")
    ]
    return " | ".join(names)


def fetch_cast(movie):
    details = get_movie_details(int(movie["tmdb_id"])) or {}
    return int(movie["tmdb_id"]), extract_cast(details)


def update_cast(engine, tmdb_id, cast_top):
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw.movies
                SET cast_top = :cast_top
                WHERE tmdb_id = :tmdb_id
                """
            ),
            {"tmdb_id": tmdb_id, "cast_top": cast_top},
        )


def coverage(engine):
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE tmdb_id IS NOT NULL) AS total,
                    COUNT(*) FILTER (
                        WHERE tmdb_id IS NOT NULL
                          AND cast_top IS NOT NULL
                          AND btrim(cast_top) <> ''
                    ) AS with_cast
                FROM raw.movies
                """
            )
        ).mappings().one()

    total = int(row["total"] or 0)
    with_cast = int(row["with_cast"] or 0)
    pct = (with_cast / total * 100) if total else 0.0
    return total, with_cast, pct


def main():
    database_url = get_database_url()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    engine = create_engine(database_url, pool_pre_ping=True)

    movies = load_missing_movies(engine)
    print(f"Movies missing cast: {len(movies)}")

    updated = 0
    no_cast = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_cast, movie): movie
            for movie in movies
        }

        for index, future in enumerate(as_completed(futures), start=1):
            movie = futures[future]
            title = movie.get("title") or str(movie["tmdb_id"])

            try:
                tmdb_id, cast_top = future.result()

                if cast_top:
                    update_cast(engine, tmdb_id, cast_top)
                    updated += 1
                    status = "updated"
                else:
                    no_cast += 1
                    status = "no cast returned"

            except Exception as exc:
                failed += 1
                status = f"failed: {exc}"

            print(f"[{index}/{len(movies)}] {title} - {status}")

    total, with_cast, pct = coverage(engine)

    print()
    print(f"Updated: {updated}")
    print(f"No cast returned: {no_cast}")
    print(f"Failed: {failed}")
    print(f"Actor cache coverage: {with_cast}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
