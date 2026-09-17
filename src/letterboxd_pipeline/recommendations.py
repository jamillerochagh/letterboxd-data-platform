import pandas as pd

from .config import LIKED_BOOST, MIN_RATING, WEIGHTS


# Genre order still matters, but genres are learned together instead of as
# completely separate "primary" and "secondary" categories.
#
# Example:
#   Drama / Romance / Comedy
# contributes:
#   Drama   = 1.00
#   Romance = 0.65
#   Comedy  = 0.35
GENRE_POSITION_WEIGHTS = {
    "genre_primary": 1.00,
    "genre_secondary": 0.65,
    "genre_tertiary": 0.35,
}

GENRE_TOTAL_WEIGHT = (
    float(WEIGHTS.get("genre_primary", 0.0))
    + float(WEIGHTS.get("genre_secondary", 0.0))
)



# =========================================================
# USER PREFERENCES
# =========================================================

def attach_user_preferences(
    enriched,
    ratings,
    likes,
):
    result = enriched.copy()

    if not ratings.empty:
        rating_columns = [
            column
            for column in [
                "Letterboxd URI",
                "Rating",
            ]
            if column in ratings.columns
        ]

        rating_data = ratings[rating_columns].copy()

        if "Rating" in rating_data.columns:
            rating_data["Rating"] = pd.to_numeric(
                rating_data["Rating"],
                errors="coerce",
            )
            rating_data = rating_data.rename(
                columns={"Rating": "rating"}
            )

        if "Letterboxd URI" in rating_data.columns:
            rating_data = rating_data.drop_duplicates(
                subset=["Letterboxd URI"],
                keep="last",
            )

            result = result.merge(
                rating_data,
                on="Letterboxd URI",
                how="left",
            )

    if "rating" not in result.columns:
        result["rating"] = pd.NA

    liked_uris = set()

    if (
        not likes.empty
        and "Letterboxd URI" in likes.columns
    ):
        liked_uris = set(
            likes["Letterboxd URI"]
            .dropna()
            .astype(str)
        )

    if "Letterboxd URI" in result.columns:
        result["liked"] = (
            result["Letterboxd URI"]
            .astype(str)
            .isin(liked_uris)
        )
    else:
        result["liked"] = False

    return result


# =========================================================
# DECADE
# =========================================================

def add_decade(df):
    result = df.copy()

    if "Year" in result.columns:
        years = pd.to_numeric(
            result["Year"],
            errors="coerce",
        )
    else:
        years = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Float64",
        )

    result["decade"] = (
        (years // 10) * 10
    ).astype("Int64")

    return result


# =========================================================
# TASTE PROFILE
# =========================================================

def _build_combined_genre_profile(enjoyed):
    """
    Learn genres as one combined preference distribution.

    A genre can contribute regardless of whether TMDB placed it first,
    second, or third. Position still matters through a decreasing weight.
    """
    genre_scores = {}

    for _, row in enjoyed.iterrows():
        movie_weight = float(
            row.get("preference_weight", 1.0)
        )

        for column, position_weight in GENRE_POSITION_WEIGHTS.items():
            value = row.get(column)

            if pd.isna(value):
                continue

            genre = str(value).strip()

            if not genre or genre.lower() == "nan":
                continue

            genre_scores[genre] = (
                genre_scores.get(genre, 0.0)
                + movie_weight * position_weight
            )

    total = sum(genre_scores.values())

    if total <= 0:
        return {}

    return {
        genre: score / total
        for genre, score in sorted(
            genre_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def build_taste_profile(df):
    """
    Build a normalized taste distribution.

    This intentionally follows the original recommender:
    - rating >= MIN_RATING is positive evidence
    - a Letterboxd like is also positive evidence
    - rating weight grows from 0.5 to 5.0
    - likes receive LIKED_BOOST
    - each attribute is normalized by the TOTAL weight,
      not by the strongest item

    Normalizing by total weight is important because it
    preserves how concentrated or broad a user's taste is.
    """
    data = add_decade(df)

    if "rating" not in data.columns:
        data["rating"] = pd.NA

    if "liked" not in data.columns:
        data["liked"] = False

    ratings = pd.to_numeric(
        data["rating"],
        errors="coerce",
    )

    liked = (
        data["liked"]
        .fillna(False)
        .astype(bool)
    )

    enjoyed = data[
        (ratings >= MIN_RATING)
        | liked
    ].copy()

    if enjoyed.empty:
        enjoyed = data.copy()

    enjoyed_ratings = pd.to_numeric(
        enjoyed["rating"],
        errors="coerce",
    )

    # Original v3 formula:
    # 3.5 -> 0.667, 4.0 -> 0.778, 4.5 -> 0.889, 5.0 -> 1.0
    enjoyed["preference_weight"] = 1.0

    rated_mask = enjoyed_ratings.notna()

    enjoyed.loc[
        rated_mask,
        "preference_weight",
    ] = (
        (
            enjoyed_ratings[rated_mask]
            .clip(lower=0.5, upper=5.0)
            - 0.5
        )
        / 4.5
    )

    enjoyed.loc[
        enjoyed["liked"].fillna(False).astype(bool),
        "preference_weight",
    ] *= LIKED_BOOST

    profile = {
        "genre_combined": _build_combined_genre_profile(
            enjoyed
        )
    }

    for attribute in WEIGHTS:
        if attribute not in enjoyed.columns:
            profile[attribute] = {}
            continue

        valid = enjoyed[
            enjoyed[attribute].notna()
        ].copy()

        valid = valid[
            valid[attribute]
            .astype(str)
            .str.strip()
            .ne("")
        ]

        if valid.empty:
            profile[attribute] = {}
            continue

        scores = (
            valid
            .groupby(attribute)["preference_weight"]
            .sum()
            .sort_values(ascending=False)
        )

        total = float(scores.sum())

        if total <= 0:
            profile[attribute] = {}
            continue

        profile[attribute] = (
            scores / total
        ).to_dict()

    return profile


# =========================================================
# SCORE ONE MOVIE
# =========================================================

def _max_attainable_profile_score(profile):
    """
    Maximum weighted content score attainable for this specific profile.

    Because preference distributions are normalized, a raw score such as 0.28
    can actually be very strong. This denominator converts the raw similarity
    into an intuitive 0-100 scale where 100 means matching the user's strongest
    learned preference in every available dimension.
    """
    maximum = 0.0

    genre_profile = profile.get(
        "genre_combined",
        {},
    )

    if genre_profile and GENRE_TOTAL_WEIGHT > 0:
        maximum += (
            max(genre_profile.values())
            * GENRE_TOTAL_WEIGHT
        )

    for attribute, attribute_weight in WEIGHTS.items():
        if attribute in (
            "genre_primary",
            "genre_secondary",
        ):
            continue

        values = profile.get(
            attribute,
            {},
        )

        if values:
            maximum += (
                max(values.values())
                * attribute_weight
            )

    return maximum


def score_movie(
    movie,
    profile,
):
    """
    Personalized content-match score from 0 to 100.

    The underlying similarity logic is unchanged, but the displayed score is
    calibrated against the strongest match theoretically attainable for this
    user's own taste profile. This keeps the ranking personalized while making
    values such as 70-90 meaningful instead of showing strong matches as 20-30.
    """
    weighted_score = 0.0

    genre_profile = profile.get(
        "genre_combined",
        {},
    )

    if genre_profile and GENRE_TOTAL_WEIGHT > 0:
        genre_match = 0.0
        genre_position_total = 0.0

        for column, position_weight in GENRE_POSITION_WEIGHTS.items():
            value = movie.get(column)

            if pd.isna(value):
                continue

            genre = str(value).strip()

            if not genre or genre.lower() == "nan":
                continue

            genre_match += (
                float(
                    genre_profile.get(
                        genre,
                        0.0,
                    )
                )
                * position_weight
            )

            genre_position_total += position_weight

        if genre_position_total > 0:
            genre_match /= genre_position_total

            weighted_score += (
                genre_match
                * GENRE_TOTAL_WEIGHT
            )

    for attribute, attribute_weight in WEIGHTS.items():
        if attribute in (
            "genre_primary",
            "genre_secondary",
        ):
            continue

        attribute_profile = profile.get(
            attribute,
            {},
        )

        if not attribute_profile:
            continue

        value = movie.get(attribute)

        if pd.isna(value):
            continue

        if attribute == "decade":
            try:
                value = int(float(value))
            except (TypeError, ValueError):
                continue
        else:
            value = str(value).strip()

            if not value or value.lower() == "nan":
                continue

        weighted_score += (
            float(
                attribute_profile.get(
                    value,
                    0.0,
                )
            )
            * attribute_weight
        )

    maximum = _max_attainable_profile_score(
        profile
    )

    if maximum <= 0:
        return 0.0

    return round(
        min(
            100.0,
            (weighted_score / maximum) * 100,
        ),
        2,
    )


# =========================================================
# RANK CANDIDATES
# =========================================================

def rank_candidates(
    candidates,
    profile,
    watched_tmdb_ids,
    limit=50,
):
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    result = add_decade(candidates)

    if "tmdb_id" in result.columns:
        ids = pd.to_numeric(
            result["tmdb_id"],
            errors="coerce",
        )

        result = result[
            ~ids.isin(watched_tmdb_ids)
        ].copy()

    result["taste_match_score"] = result.apply(
        lambda row: score_movie(
            row,
            profile,
        ),
        axis=1,
    )

    result = result[
        result["taste_match_score"] > 0
    ].copy()

    # Quality is deliberately NOT part of the taste score.
    # It is only a late tie-breaker, so individual taste
    # remains the dominant ranking signal.
    result["_quality_tiebreak"] = pd.to_numeric(
        result.get(
            "vote_average",
            pd.Series(
                0,
                index=result.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    result = result.sort_values(
        [
            "taste_match_score",
            "_quality_tiebreak",
        ],
        ascending=[
            False,
            False,
        ],
    )

    if "tmdb_id" in result.columns:
        result = result.drop_duplicates(
            subset=["tmdb_id"]
        )

    result = result.drop(
        columns=["_quality_tiebreak"],
        errors="ignore",
    )

    return (
        result
        .head(limit)
        .reset_index(drop=True)
    )


# =========================================================
# PROFILE SUMMARY
# =========================================================

def get_profile_summary(profile):
    summary = {}

    for attribute, values in profile.items():
        if not values:
            summary[attribute] = None
            continue

        summary[attribute] = max(
            values,
            key=values.get,
        )

    # Existing UI expects genre_primary as the displayed favorite genre.
    if profile.get("genre_combined"):
        summary["genre_primary"] = max(
            profile["genre_combined"],
            key=profile["genre_combined"].get,
        )

    return summary
