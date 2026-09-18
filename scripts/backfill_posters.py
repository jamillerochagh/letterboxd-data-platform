"""
Populate raw.movies.poster_path from TMDB without making the Streamlit UI slow.

Run from the project root:
    source .venv/bin/activate
    PYTHONPATH=. python scripts/backfill_posters.py
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import create_engine, text

from src.letterboxd_pipeline.database import get_database_url
from src.letterboxd_pipeline.tmdb import get_movie_details


MAX_WORKERS = 6
BATCH_LIMIT = 5000


def load_missing(engine):
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT tmdb_id, title, release_year
                FROM raw.movies
                WHERE tmdb_id IS NOT NULL
                  AND (poster_path IS NULL OR btrim(poster_path) = '')
                ORDER BY tmdb_id
                LIMIT :limit
                """
            ),
            {"limit": BATCH_LIMIT},
        )
        return [dict(row._mapping) for row in rows]


def fetch_poster(movie):
    details = get_movie_details(int(movie["tmdb_id"])) or {}
    return int(movie["tmdb_id"]), (details.get("poster_path") or "").strip()


def update_poster(engine, tmdb_id, poster_path):
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw.movies
                SET poster_path = :poster_path
                WHERE tmdb_id = :tmdb_id
                """
            ),
            {"tmdb_id": tmdb_id, "poster_path": poster_path},
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
                          AND poster_path IS NOT NULL
                          AND btrim(poster_path) <> ''
                    ) AS with_poster
                FROM raw.movies
                """
            )
        ).mappings().one()

    total = int(row["total"] or 0)
    with_poster = int(row["with_poster"] or 0)
    pct = (with_poster / total * 100) if total else 0.0
    return total, with_poster, pct


def main():
    database_url = get_database_url()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    engine = create_engine(database_url, pool_pre_ping=True)
    movies = load_missing(engine)

    print(f"Movies missing posters: {len(movies)}")

    updated = 0
    no_poster = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_poster, movie): movie
            for movie in movies
        }

        for index, future in enumerate(as_completed(futures), start=1):
            movie = futures[future]
            title = movie.get("title") or str(movie["tmdb_id"])

            try:
                tmdb_id, poster_path = future.result()
                if poster_path:
                    update_poster(engine, tmdb_id, poster_path)
                    updated += 1
                    status = "updated"
                else:
                    no_poster += 1
                    status = "no poster returned"
            except Exception as exc:
                failed += 1
                status = f"failed: {exc}"

            print(f"[{index}/{len(movies)}] {title} - {status}")

    total, with_poster, pct = coverage(engine)
    print()
    print(f"Updated: {updated}")
    print(f"No poster returned: {no_poster}")
    print(f"Failed: {failed}")
    print(f"Poster cache coverage: {with_poster}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
