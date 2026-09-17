from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from .config import TMDB_API_KEY, TMDB_BASE_URL


def _request(endpoint: str, params: dict) -> dict:
    """Send a request to the TMDB API."""
    params = {
        **params,
        "api_key": TMDB_API_KEY,
    }

    response = requests.get(
        f"{TMDB_BASE_URL}{endpoint}",
        params=params,
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def search_movie(
    title: str,
    year: str = "",
) -> Optional[dict]:
    """Search TMDB for a movie using title and optional year."""
    params = {
        "query": title,
        "language": "en-US",
    }

    if year:
        params["year"] = year

    data = _request(
        "/search/movie",
        params,
    )

    results = data.get("results", [])

    if not results and year:
        params.pop("year")

        data = _request(
            "/search/movie",
            params,
        )

        results = data.get("results", [])

    return results[0] if results else None


def get_movie_details(
    movie_id: int,
) -> dict:
    """Get detailed movie metadata and credits."""
    return _request(
        f"/movie/{movie_id}",
        {
            "append_to_response": "credits",
            "language": "en-US",
        },
    )


def empty_movie_metadata() -> dict:
    return {
        "tmdb_id": "",
        "director": "",
        "genre_primary": "",
        "genre_secondary": "",
        "genre_tertiary": "",
        "country_primary": "",
        "original_language": "",
        "runtime_min": "",
        "vote_average": "",
        "popularity": "",
        "poster_path": "",
        "tagline": "",
        "overview": "",
    }


def fetch_movie_metadata(
    title: str,
    year: str = "",
) -> dict:
    """Search for a movie and return normalized TMDB metadata."""
    result = search_movie(
        title,
        year,
    )

    if not result:
        return empty_movie_metadata()

    details = get_movie_details(
        result["id"]
    )

    credits = details.get(
        "credits",
        {},
    )

    directors = [
        person["name"]
        for person in credits.get(
            "crew",
            [],
        )
        if person.get("job") == "Director"
    ]

    genres = [
        genre["name"]
        for genre in details.get(
            "genres",
            [],
        )
    ]

    countries = [
        country["name"]
        for country in details.get(
            "production_countries",
            [],
        )
    ]

    return {
        "tmdb_id": details.get("id", ""),
        "director": directors[0] if directors else "",
        "genre_primary": genres[0] if len(genres) > 0 else "",
        "genre_secondary": genres[1] if len(genres) > 1 else "",
        "genre_tertiary": genres[2] if len(genres) > 2 else "",
        "country_primary": countries[0] if countries else "",
        "original_language": details.get(
            "original_language",
            "",
        ),
        "runtime_min": details.get("runtime", ""),
        "vote_average": details.get(
            "vote_average",
            "",
        ),
        "popularity": details.get(
            "popularity",
            "",
        ),
        "poster_path": details.get(
            "poster_path",
            "",
        ),
        "tagline": details.get(
            "tagline",
            "",
        ),
        "overview": details.get(
            "overview",
            "",
        ),
    }


GENRE_IDS = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37,
}

GENRE_ID_TO_NAME = {
    value: key
    for key, value in GENRE_IDS.items()
}


def discover_movies(
    genre_id: int | None = None,
    page: int = 1,
    sort_by: str = "vote_count.desc",
) -> list[dict]:
    """
    Discover established movies from TMDB.

    vote_count.desc is intentional: the previous
    vote_average.desc strategy repeatedly surfaced the
    same highly-rated titles for different users.
    """
    params = {
        "language": "en-US",
        "sort_by": sort_by,
        "vote_count.gte": 100,
        "page": page,
        "include_adult": False,
    }

    if genre_id:
        params["with_genres"] = genre_id

    data = _request(
        "/discover/movie",
        params,
    )

    return data.get("results", [])


def _discover_primary_genre(
    movie: dict,
) -> str:
    genre_ids = movie.get(
        "genre_ids",
        [],
    ) or []

    if not genre_ids:
        return ""

    return GENRE_ID_TO_NAME.get(
        genre_ids[0],
        "",
    )


def build_candidate_catalog(
    preferred_genres: list[str],
    pages_per_genre: int = 4,
) -> list[dict]:
    """
    Build a broad but still practical candidate pool.

    Changes from the first Streamlit version:
    - use the user's TOP 8 primary genres, not only 5
    - use more pages per genre
    - sort by vote count rather than vote average
    - deduplicate globally
    - keep movies whose primary genre belongs to the
      user's preferred set when that information exists

    The goal is to give the personalized scoring stage
    a much richer pool without fetching the entire TMDB.
    """
    preferred = [
        genre
        for genre in preferred_genres
        if genre in GENRE_IDS
    ][:8]

    if not preferred:
        preferred = list(
            GENRE_IDS.keys()
        )[:8]

    preferred_set = set(preferred)
    candidates = {}

    for genre in preferred:
        genre_id = GENRE_IDS[genre]

        for page in range(
            1,
            pages_per_genre + 1,
        ):
            movies = discover_movies(
                genre_id=genre_id,
                page=page,
                sort_by="vote_count.desc",
            )

            for movie in movies:
                movie_id = movie.get("id")

                if not movie_id:
                    continue

                primary_genre = (
                    _discover_primary_genre(
                        movie
                    )
                )

                # If TMDB gives us a primary genre, prefer
                # candidates that are genuinely central to
                # the user's taste rather than merely having
                # the searched genre somewhere in the list.
                if (
                    primary_genre
                    and primary_genre
                    not in preferred_set
                ):
                    continue

                candidates[movie_id] = movie

    return list(candidates.values())


def enrich_candidate_catalog(candidates: list[dict], max_workers: int = 8) -> pd.DataFrame:
    """Enrich candidates concurrently and preserve discovery posters as fallback."""
    if not candidates:
        return pd.DataFrame()

    def enrich_one(candidate):
        movie_id = candidate.get("id")
        if not movie_id:
            return None
        try:
            details = get_movie_details(movie_id)
        except requests.RequestException:
            return None
        credits = details.get("credits", {}) or {}
        directors = [p.get("name", "") for p in credits.get("crew", []) if p.get("job") == "Director"]
        genres = [g.get("name", "") for g in details.get("genres", []) if g.get("name")]
        countries = [c.get("name", "") for c in details.get("production_countries", []) if c.get("name")]
        release_date = details.get("release_date") or candidate.get("release_date") or ""
        release_year = None
        if release_date:
            try:
                release_year = int(str(release_date)[:4])
            except (TypeError, ValueError):
                pass
        return {
            "tmdb_id": movie_id, "title": details.get("title") or candidate.get("title", ""),
            "Year": release_year, "director": directors[0] if directors else "",
            "genre_primary": genres[0] if len(genres) > 0 else "",
            "genre_secondary": genres[1] if len(genres) > 1 else "",
            "genre_tertiary": genres[2] if len(genres) > 2 else "",
            "country_primary": countries[0] if countries else "",
            "original_language": details.get("original_language") or candidate.get("original_language", ""),
            "runtime_min": details.get("runtime"),
            "vote_average": details.get("vote_average") if details.get("vote_average") is not None else candidate.get("vote_average"),
            "popularity": details.get("popularity") if details.get("popularity") is not None else candidate.get("popularity"),
            "poster_path": details.get("poster_path") or candidate.get("poster_path") or "",
            "overview": details.get("overview") or candidate.get("overview") or "",
        }

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 8))) as executor:
        futures = [executor.submit(enrich_one, c) for c in candidates]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception:
                row = None
            if row:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["tmdb_id"]).reset_index(drop=True)