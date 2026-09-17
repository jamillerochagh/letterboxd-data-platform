import pandas as pd

from .recommendations import (
    add_decade,
    build_taste_profile,
    score_movie,
)


# =========================================================
# HELPERS
# =========================================================

def _normalize_tmdb_ids(df: pd.DataFrame) -> set[int]:
    """Return valid TMDB IDs from a movie DataFrame."""
    if df is None or df.empty or "tmdb_id" not in df.columns:
        return set()

    return set(
        pd.to_numeric(
            df["tmdb_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )


def _movie_key(row: pd.Series) -> str:
    """
    Build a fallback movie identifier when TMDB ID is missing.
    """
    tmdb_id = row.get("tmdb_id")

    if pd.notna(tmdb_id):
        try:
            return f"tmdb:{int(float(tmdb_id))}"
        except (TypeError, ValueError):
            pass

    title = str(
        row.get(
            "Name",
            row.get("title", ""),
        )
    ).strip().lower()

    year = str(
        row.get(
            "Year",
            row.get("release_year", ""),
        )
    ).strip()

    if year.endswith(".0"):
        year = year[:-2]

    return f"title:{title}|year:{year}"


def _prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize movie history before comparing two users."""
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    result["_movie_key"] = result.apply(
        _movie_key,
        axis=1,
    )

    if "rating" not in result.columns:
        result["rating"] = pd.NA

    result["rating"] = pd.to_numeric(
        result["rating"],
        errors="coerce",
    )

    if "liked" not in result.columns:
        result["liked"] = False

    result["liked"] = (
        result["liked"]
        .fillna(False)
        .astype(bool)
    )

    return result


def _display_title(row: pd.Series) -> str:
    """Return the best available movie title."""
    for column in ["Name", "title"]:
        value = row.get(column)

        if pd.notna(value):
            value = str(value).strip()

            if value:
                return value

    return "Unknown movie"


# =========================================================
# MOVIES IN COMMON
# =========================================================

def get_movies_in_common(
    creator_history: pd.DataFrame,
    friend_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return movies watched by both users together with
    their ratings and rating difference.
    """
    creator = _prepare_history(
        creator_history
    )

    friend = _prepare_history(
        friend_history
    )

    if creator.empty or friend.empty:
        return pd.DataFrame()

    creator = creator.drop_duplicates(
        subset=["_movie_key"],
        keep="last",
    )

    friend = friend.drop_duplicates(
        subset=["_movie_key"],
        keep="last",
    )

    common_keys = set(
        creator["_movie_key"]
    ).intersection(
        friend["_movie_key"]
    )

    rows = []

    for key in common_keys:
        creator_row = creator[
            creator["_movie_key"] == key
        ].iloc[0]

        friend_row = friend[
            friend["_movie_key"] == key
        ].iloc[0]

        creator_rating = creator_row.get(
            "rating"
        )

        friend_rating = friend_row.get(
            "rating"
        )

        rating_gap = pd.NA

        if (
            pd.notna(creator_rating)
            and pd.notna(friend_rating)
        ):
            rating_gap = abs(
                float(creator_rating)
                - float(friend_rating)
            )

        rows.append(
            {
                "movie_key": key,
                "tmdb_id": creator_row.get(
                    "tmdb_id"
                ),
                "title": _display_title(
                    creator_row
                ),
                "year": creator_row.get(
                    "Year",
                    creator_row.get(
                        "release_year"
                    ),
                ),
                "poster_path": (
                    creator_row.get("poster_path")
                    if pd.notna(creator_row.get("poster_path"))
                    and str(creator_row.get("poster_path")).strip()
                    else friend_row.get("poster_path")
                ),
                "creator_rating": creator_rating,
                "friend_rating": friend_rating,
                "creator_liked": creator_row.get(
                    "liked",
                    False,
                ),
                "friend_liked": friend_row.get(
                    "liked",
                    False,
                ),
                "rating_gap": rating_gap,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        ["rating_gap", "title"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


# =========================================================
# RATING COMPATIBILITY
# =========================================================

def calculate_average_rating_gap(
    common_movies: pd.DataFrame,
) -> float | None:
    """Average rating difference for commonly rated movies."""
    if (
        common_movies is None
        or common_movies.empty
        or "rating_gap" not in common_movies.columns
    ):
        return None

    gaps = pd.to_numeric(
        common_movies["rating_gap"],
        errors="coerce",
    ).dropna()

    if gaps.empty:
        return None

    return round(
        float(gaps.mean()),
        2,
    )


def calculate_rating_match(
    common_movies: pd.DataFrame,
) -> float | None:
    """
    Convert rating agreement into a 0–100 compatibility score.

    A 0-star difference = 100.
    A 5-star difference = 0.
    """
    average_gap = calculate_average_rating_gap(
        common_movies
    )

    if average_gap is None:
        return None

    score = (
        1 - min(
            average_gap / 5.0,
            1.0,
        )
    ) * 100

    return round(score, 1)


# =========================================================
# PROFILE SIMILARITY
# =========================================================

def calculate_attribute_similarity(
    creator_values: dict,
    friend_values: dict,
) -> float | None:
    """
    Compare one taste-profile attribute.

    Profiles contain values normalized from 0 to 1.
    Similarity is based on the overlap between both profiles.
    """
    creator_values = creator_values or {}
    friend_values = friend_values or {}

    if not creator_values or not friend_values:
        return None

    all_values = set(
        creator_values
    ).union(
        friend_values
    )

    if not all_values:
        return None

    overlap = 0.0
    total = 0.0

    for value in all_values:
        creator_score = float(
            creator_values.get(
                value,
                0.0,
            )
        )

        friend_score = float(
            friend_values.get(
                value,
                0.0,
            )
        )

        overlap += min(
            creator_score,
            friend_score,
        )

        total += max(
            creator_score,
            friend_score,
        )

    if total == 0:
        return None

    return overlap / total


def calculate_profile_similarity(
    creator_profile: dict,
    friend_profile: dict,
) -> tuple[float, dict]:
    """
    Calculate overall taste-profile similarity.

    Only attributes available for both users contribute.
    """
    attributes = set(
        creator_profile
    ).union(
        friend_profile
    )

    attribute_scores = {}

    for attribute in attributes:
        similarity = calculate_attribute_similarity(
            creator_profile.get(
                attribute,
                {},
            ),
            friend_profile.get(
                attribute,
                {},
            ),
        )

        if similarity is not None:
            attribute_scores[
                attribute
            ] = round(
                similarity * 100,
                1,
            )

    if not attribute_scores:
        return 0.0, {}

    overall = sum(
        attribute_scores.values()
    ) / len(attribute_scores)

    return (
        round(overall, 1),
        attribute_scores,
    )


# =========================================================
# FINAL TASTE MATCH
# =========================================================

def calculate_taste_match(
    creator_profile: dict,
    friend_profile: dict,
    common_movies: pd.DataFrame,
) -> float:
    """
    Calculate the headline Movie Blend score.

    Profile similarity is the main signal.
    Rating agreement adds extra evidence when both users
    have rated movies in common.
    """
    profile_match, _ = (
        calculate_profile_similarity(
            creator_profile,
            friend_profile,
        )
    )

    rating_match = calculate_rating_match(
        common_movies
    )

    if rating_match is None:
        return profile_match

    # Taste preferences are the main signal.
    # Direct rating agreement is secondary evidence.

    final_score = (
        profile_match * 0.75
        + rating_match * 0.25
    )

    return round(
        final_score,
        1,
    )


# =========================================================
# SHARED PREFERENCES
# =========================================================

def get_shared_top_preference(
    creator_profile: dict,
    friend_profile: dict,
    attribute: str,
):
    """
    Return the strongest shared preference for an attribute.
    """
    creator_values = creator_profile.get(
        attribute,
        {},
    )

    friend_values = friend_profile.get(
        attribute,
        {},
    )

    shared = set(
        creator_values
    ).intersection(
        friend_values
    )

    if not shared:
        return None

    shared_scores = {
        value: (
            float(creator_values[value])
            + float(friend_values[value])
        ) / 2
        for value in shared
    }

    return max(
        shared_scores,
        key=shared_scores.get,
    )


# =========================================================
# AGREEMENTS
# =========================================================

def get_movies_both_love(
    common_movies: pd.DataFrame,
    limit: int = 50,
) -> pd.DataFrame:
    """Return movies both users rated exactly 5.0."""
    if common_movies is None or common_movies.empty:
        return pd.DataFrame()

    result = common_movies.copy()

    creator_rating = pd.to_numeric(
        result["creator_rating"],
        errors="coerce",
    )

    friend_rating = pd.to_numeric(
        result["friend_rating"],
        errors="coerce",
    )

    result = result[
        creator_rating.eq(5.0)
        & friend_rating.eq(5.0)
    ].copy()

    if result.empty:
        return result

    return (
        result
        .sort_values("title")
        .head(limit)
        .reset_index(drop=True)
    )

def get_shared_preferences(
    creator_profile: dict,
    friend_profile: dict,
) -> dict:
    """
    Return the strongest shared preference for each
    relevant taste attribute.
    """

    attributes = [
        "genre_primary",
        "genre_secondary",
        "director",
        "country_primary",
        "original_language",
        "decade",
    ]

    shared_preferences = {}

    for attribute in attributes:

        value = get_shared_top_preference(
            creator_profile,
            friend_profile,
            attribute,
        )

        if value is not None:
            shared_preferences[
                attribute
            ] = value

    return shared_preferences


# =========================================================
# DISAGREEMENTS
# =========================================================

def get_biggest_disagreements(
    common_movies: pd.DataFrame,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Return commonly watched movies with the largest
    difference between the users' ratings.
    """
    if common_movies is None or common_movies.empty:
        return pd.DataFrame()

    result = common_movies.copy()

    result["rating_gap"] = pd.to_numeric(
        result["rating_gap"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["rating_gap"]
    )

    return (
        result
        .sort_values(
            "rating_gap",
            ascending=False,
        )
        .head(limit)
        .reset_index(drop=True)
    )


# =========================================================
# COMBINED PROFILE
# =========================================================

def build_blended_profile(
    creator_profile: dict,
    friend_profile: dict,
) -> dict:
    """
    Combine two taste profiles.

    A movie characteristic must appeal to both users
    to receive a strong Blend preference.
    """
    blended = {}

    attributes = set(
        creator_profile
    ).union(
        friend_profile
    )

    for attribute in attributes:
        creator_values = creator_profile.get(
            attribute,
            {},
        )

        friend_values = friend_profile.get(
            attribute,
            {},
        )

        all_values = set(
            creator_values
        ).union(
            friend_values
        )

        scores = {}

        for value in all_values:
            creator_score = float(
                creator_values.get(
                    value,
                    0.0,
                )
            )

            friend_score = float(
                friend_values.get(
                    value,
                    0.0,
                )
            )

            # Geometric mean rewards characteristics
            # liked by BOTH users and strongly penalizes
            # one-sided preferences.

            if (
                creator_score > 0
                and friend_score > 0
            ):
                combined_score = (
                    creator_score
                    * friend_score
                ) ** 0.5
            else:
                combined_score = 0.0

            if combined_score > 0:
                scores[value] = round(
                    combined_score,
                    4,
                )

        if scores:
            maximum = max(
                scores.values()
            )

            if maximum > 0:
                scores = {
                    value: round(
                        score / maximum,
                        4,
                    )
                    for value, score
                    in scores.items()
                }

        blended[attribute] = scores

    return blended


# =========================================================
# WHAT SHOULD WE WATCH?
# =========================================================

def rank_blend_candidates(
    candidates: pd.DataFrame,
    creator_profile: dict,
    friend_profile: dict,
    creator_history: pd.DataFrame,
    friend_history: pd.DataFrame,
    limit: int = 30,
) -> pd.DataFrame:
    """
    Rank movies for two users.

    Movies watched by either user are removed.

    Each recommendation receives:
    - creator_match_score
    - friend_match_score
    - blend_score
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    result = add_decade(
        candidates
    )

    watched_ids = (
        _normalize_tmdb_ids(
            creator_history
        )
        | _normalize_tmdb_ids(
            friend_history
        )
    )

    if "tmdb_id" in result.columns:
        candidate_ids = pd.to_numeric(
            result["tmdb_id"],
            errors="coerce",
        )

        result = result[
            ~candidate_ids.isin(
                watched_ids
            )
        ].copy()

    result[
        "creator_match_score"
    ] = result.apply(
        lambda row: score_movie(
            row,
            creator_profile,
        ),
        axis=1,
    )

    result[
        "friend_match_score"
    ] = result.apply(
        lambda row: score_movie(
            row,
            friend_profile,
        ),
        axis=1,
    )

    creator_scores = result["creator_match_score"].astype(float)
    friend_scores = result["friend_match_score"].astype(float)
    score_sum = creator_scores + friend_scores

    harmonic_mean = (
        2.0 * creator_scores * friend_scores
        / score_sum.where(score_sum > 0, 1.0)
    )
    weaker_match = pd.concat([creator_scores, friend_scores], axis=1).min(axis=1)
    rating_balance = (100.0 - (creator_scores - friend_scores).abs()).clip(0.0, 100.0)

    # Consensus-oriented Blend: favor movies that are strong for both users,
    # not merely exceptional for one profile.
    result["blend_score"] = (
        0.55 * harmonic_mean
        + 0.35 * weaker_match
        + 0.10 * rating_balance
    ).clip(0.0, 100.0).round(1)

    result = result[
        (
            result["creator_match_score"] > 0
        )
        & (
            result["friend_match_score"] > 0
        )
    ].copy()

    sort_columns = [
        "blend_score"
    ]

    ascending = [
        False
    ]

    if "vote_average" in result.columns:
        result["vote_average"] = pd.to_numeric(
            result["vote_average"],
            errors="coerce",
        )

        sort_columns.append(
            "vote_average"
        )

        ascending.append(
            False
        )

    result = result.sort_values(
        sort_columns,
        ascending=ascending,
    )

    if "tmdb_id" in result.columns:
        result = result.drop_duplicates(
            subset=["tmdb_id"]
        )

    return (
        result
        .head(limit)
        .reset_index(drop=True)
    )


# =========================================================
# COMPLETE BLEND ANALYSIS
# =========================================================

def build_blend_analysis(
    creator_history: pd.DataFrame,
    friend_history: pd.DataFrame,
) -> dict:
    """
    Build all non-recommendation Movie Blend analytics.
    """

    creator_profile = build_taste_profile(
        creator_history
    )

    friend_profile = build_taste_profile(
        friend_history
    )

    common_movies = get_movies_in_common(
        creator_history,
        friend_history,
    )

    profile_match, attribute_matches = (
        calculate_profile_similarity(
            creator_profile,
            friend_profile,
        )
    )

    rating_match = calculate_rating_match(
        common_movies
    )

    taste_match = calculate_taste_match(
        creator_profile,
        friend_profile,
        common_movies,
    )

    blended_profile = build_blended_profile(
        creator_profile,
        friend_profile,
    )

    return {
        "taste_match": taste_match,
        "profile_match": profile_match,
        "rating_match": rating_match,
        "movies_in_common": len(
            common_movies
        ),
        "average_rating_gap": (
            calculate_average_rating_gap(
                common_movies
            )
        ),
        "shared_top_genre": (
            get_shared_top_preference(
                creator_profile,
                friend_profile,
                "genre_primary",
            )
        ),
        "shared_top_director": (
            get_shared_top_preference(
                creator_profile,
                friend_profile,
                "director",
            )
        ),
        "attribute_matches": (
            attribute_matches
        ),
        "shared_preferences": (
            get_shared_preferences(
                creator_profile,
                friend_profile,
            )
        ),
        "creator_profile": (
            creator_profile
        ),
        "friend_profile": (
            friend_profile
        ),
        "blended_profile": (
            blended_profile
        ),
        "common_movies": (
            common_movies
        ),
        "movies_both_love": (
            get_movies_both_love(
                common_movies
            )
        ),
        "biggest_disagreements": (
            get_biggest_disagreements(
                common_movies
            )
        ),
    }