from typing import Optional

import pandas as pd
import requests

from .config import TMDB_API_KEY, TMDB_BASE_URL


def _request(endpoint: str, params: dict) -> dict:
    """Send a request to the TMDB API."""

    params = {
        **params,
        "api_key": TMDB_API_KEY,
    }

    print(
    "TMDB CONFIG | "
    f"key_loaded={bool(TMDB_API_KEY)} | "
    f"key_length={len(TMDB_API_KEY) if TMDB_API_KEY else 0}"
)

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
    """Search TMDB for a movie using title and, when available, year."""

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

    results = data.get(
        "results",
        [],
    )

    # Retry without year if TMDB did not find a match.
    if not results and year:
        params.pop("year")

        data = _request(
            "/search/movie",
            params,
        )

        results = data.get(
            "results",
            [],
        )

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
    """Return the expected schema when TMDB has no match."""

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
        "tmdb_id": details.get(
            "id",
            "",
        ),
        "director": (
            directors[0]
            if directors
            else ""
        ),
        "genre_primary": (
            genres[0]
            if len(genres) > 0
            else ""
        ),
        "genre_secondary": (
            genres[1]
            if len(genres) > 1
            else ""
        ),
        "genre_tertiary": (
            genres[2]
            if len(genres) > 2
            else ""
        ),
        "country_primary": (
            countries[0]
            if countries
            else ""
        ),
        "original_language": details.get(
            "original_language",
            "",
        ),
        "runtime_min": details.get(
            "runtime",
            "",
        ),
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
            ""
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


def discover_movies(
    genre_id: int | None = None,
    page: int = 1,
) -> list[dict]:
    """Discover well-rated movies from TMDB."""

    params = {
        "language": "en-US",
        "sort_by": "vote_average.desc",
        "vote_count.gte": 200,
        "page": page,
    }

    if genre_id:
        params["with_genres"] = genre_id

    data = _request(
        "/discover/movie",
        params,
    )

    return data.get(
        "results",
        [],
    )


def build_candidate_catalog(
    preferred_genres: list[str],
    pages_per_genre: int = 2,
) -> list[dict]:
    """
    Build a recommendation candidate catalog
    from the user's strongest genres.
    """

    candidates = {}

    genres_to_search = preferred_genres[:5]

    for genre in genres_to_search:
        genre_id = GENRE_IDS.get(
            genre
        )

        if not genre_id:
            continue

        for page in range(
            1,
            pages_per_genre + 1,
        ):
            movies = discover_movies(
                genre_id=genre_id,
                page=page,
            )

            for movie in movies:
                movie_id = movie.get(
                    "id"
                )

                if movie_id:
                    candidates[
                        movie_id
                    ] = movie

    return list(
        candidates.values()
    )


def enrich_candidate_catalog(
    candidates: list[dict],
) -> pd.DataFrame:
    """
    Convert TMDB discovery results into the same
    metadata format used by the recommendation engine.
    """

    rows = []

    for candidate in candidates:
        movie_id = candidate.get(
            "id"
        )

        if not movie_id:
            continue

        try:
            details = get_movie_details(
                movie_id
            )
        except requests.RequestException:
            continue

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

        release_date = details.get(
            "release_date",
            "",
        )

        release_year = None

        if release_date:
            try:
                release_year = int(
                    release_date[:4]
                )
            except ValueError:
                pass

        rows.append(
            {
                "tmdb_id": movie_id,
                "title": details.get(
                    "title",
                    "",
                ),
                "Year": release_year,
                "director": (
                    directors[0]
                    if directors
                    else ""
                ),
                "genre_primary": (
                    genres[0]
                    if len(genres) > 0
                    else ""
                ),
                "genre_secondary": (
                    genres[1]
                    if len(genres) > 1
                    else ""
                ),
                "genre_tertiary": (
                    genres[2]
                    if len(genres) > 2
                    else ""
                ),
                "country_primary": (
                    countries[0]
                    if countries
                    else ""
                ),
                "original_language": details.get(
                    "original_language",
                    "",
                ),
                "runtime_min": details.get(
                    "runtime",
                ),
                "vote_average": details.get(
                    "vote_average",
                ),
                "popularity": details.get(
                    "popularity",
                ),
                "poster_path": details.get(
                    "poster_path",
                    ""
                ),
                "overview": details.get(
                    "overview",
                    "",
                ),
            }
        )

    return pd.DataFrame(
        rows
    )