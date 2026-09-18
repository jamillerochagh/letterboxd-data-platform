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

def _normalize_uri(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.rstrip("/")
    )


def _movie_key(data):
    if "Name" in data.columns:
        names = data["Name"]
    elif "title" in data.columns:
        names = data["title"]
    else:
        names = pd.Series("", index=data.index)

    if "Year" in data.columns:
        years = data["Year"]
    elif "release_year" in data.columns:
        years = data["release_year"]
    else:
        years = pd.Series("", index=data.index)

    names = (
        names.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    years = (
        pd.to_numeric(years, errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    return names + "||" + years


def attach_user_preferences(
    enriched,
    ratings,
    likes,
):
    """
    Attach Letterboxd ratings/likes to enriched metadata.

    Matching priority:
    1. exact normalized Letterboxd URI
    2. normalized title + release year only for rows still unmatched

    Existing rating/liked columns from earlier merges are discarded first,
    preventing stale values from being carried into Overview or profiles.
    """
    result = enriched.copy()

    result = result.drop(
        columns=["rating", "liked"],
        errors="ignore",
    )
    result["rating"] = pd.NA
    result["liked"] = False

    rating_data = ratings.copy() if ratings is not None else pd.DataFrame()

    if not rating_data.empty and "Rating" in rating_data.columns:
        rating_data["Rating"] = pd.to_numeric(
            rating_data["Rating"],
            errors="coerce",
        )

        # URI-first: Letterboxd's own stable movie identifier.
        if (
            "Letterboxd URI" in result.columns
            and "Letterboxd URI" in rating_data.columns
        ):
            result["_lb_uri"] = _normalize_uri(result["Letterboxd URI"])
            rating_data["_lb_uri"] = _normalize_uri(rating_data["Letterboxd URI"])

            uri_ratings = (
                rating_data[
                    rating_data["_lb_uri"].ne("")
                ][["_lb_uri", "Rating"]]
                .drop_duplicates("_lb_uri", keep="last")
                .set_index("_lb_uri")["Rating"]
            )

            result["rating"] = result["_lb_uri"].map(uri_ratings)

        # Fallback only for genuinely unmatched rows.
        unmatched = result["rating"].isna()
        if unmatched.any() and {"Name", "Year"}.issubset(rating_data.columns):
            rating_data["_movie_key"] = _movie_key(rating_data)
            result["_movie_key"] = _movie_key(result)

            key_ratings = (
                rating_data[
                    rating_data["_movie_key"].ne("||<NA>")
                ][["_movie_key", "Rating"]]
                .drop_duplicates("_movie_key", keep="last")
                .set_index("_movie_key")["Rating"]
            )
            result.loc[unmatched, "rating"] = (
                result.loc[unmatched, "_movie_key"].map(key_ratings)
            )

    if likes is not None and not likes.empty:
        if (
            "Letterboxd URI" in result.columns
            and "Letterboxd URI" in likes.columns
        ):
            if "_lb_uri" not in result.columns:
                result["_lb_uri"] = _normalize_uri(result["Letterboxd URI"])

            liked_uris = set(
                _normalize_uri(likes["Letterboxd URI"])
                .loc[lambda values: values.ne("")]
            )
            result["liked"] = result["_lb_uri"].isin(liked_uris)

        # Like fallback by title/year only where URI was unavailable.
        if {"Name", "Year"}.issubset(likes.columns):
            if "_movie_key" not in result.columns:
                result["_movie_key"] = _movie_key(result)

            liked_keys = set(_movie_key(likes))
            no_uri = (
                result["_lb_uri"].eq("")
                if "_lb_uri" in result.columns
                else pd.Series(True, index=result.index)
            )
            result.loc[no_uri, "liked"] = (
                result.loc[no_uri, "_movie_key"].isin(liked_keys)
            )

    result["rating"] = pd.to_numeric(
        result["rating"],
        errors="coerce",
    )

    return result.drop(
        columns=["_lb_uri", "_movie_key"],
        errors="ignore",
    )


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

def _quality_score(value):
    """Convert TMDB's 0-10 rating to a conservative 0-100 quality signal."""
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return 0.0

    # 5.5 is effectively the floor of our usable catalogue.
    return max(0.0, min(100.0, ((rating - 5.5) / 3.5) * 100.0))


def _diversify_ranked_candidates(
    ranked,
    limit,
    genre_penalty=4.0,
    director_penalty=6.0,
):
    """
    Greedy diversity-aware re-ranking.

    High-personalization movies remain near the top, while repeated genres
    and directors receive a small penalty as the recommendation page fills.
    """
    if ranked is None or ranked.empty:
        return pd.DataFrame()

    remaining = ranked.copy()
    selected_rows = []
    genre_counts = {}
    director_counts = {}

    while not remaining.empty and len(selected_rows) < limit:
        best_index = None
        best_adjusted = None

        for index, row in remaining.iterrows():
            adjusted = float(row.get("recommendation_score", 0.0))

            genre = str(row.get("genre_primary", "") or "").strip()
            director = str(row.get("director", "") or "").strip()

            if genre:
                adjusted -= genre_penalty * genre_counts.get(genre, 0)

            if director:
                adjusted -= director_penalty * director_counts.get(director, 0)

            if best_adjusted is None or adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = index

        if best_index is None:
            break

        chosen = remaining.loc[best_index].copy()
        chosen["diversified_score"] = round(float(best_adjusted), 2)
        selected_rows.append(chosen)

        genre = str(chosen.get("genre_primary", "") or "").strip()
        director = str(chosen.get("director", "") or "").strip()

        if genre:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        if director:
            director_counts[director] = director_counts.get(director, 0) + 1

        remaining = remaining.drop(index=best_index)

    if not selected_rows:
        return pd.DataFrame()

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def rank_candidates(
    candidates,
    profile,
    watched_tmdb_ids,
    limit=50,
):
    """
    Multi-stage personalized ranking.

    1. Remove watched movies.
    2. Calculate the user's content/taste match.
    3. Apply a modest TMDB quality signal.
    4. Re-rank for genre/director diversity.

    Taste remains the dominant signal; public quality prevents mediocre,
    universally-popular candidates from floating to the top for everyone.
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    result = add_decade(candidates)

    if "tmdb_id" in result.columns:
        ids = pd.to_numeric(result["tmdb_id"], errors="coerce")
        result = result[~ids.isin(watched_tmdb_ids)].copy()

    result["taste_match_score"] = result.apply(
        lambda row: score_movie(row, profile),
        axis=1,
    )

    result["vote_average"] = pd.to_numeric(
        result.get("vote_average", pd.Series(0, index=result.index)),
        errors="coerce",
    ).fillna(0)

    result = result[
        (result["taste_match_score"] > 0)
        & (result["vote_average"] >= 6.2)
    ].copy()

    if result.empty:
        return result

    result["quality_score"] = result["vote_average"].apply(_quality_score)

    # Personal taste dominates. Quality is intentionally secondary.
    result["recommendation_score"] = (
        result["taste_match_score"] * 0.88
        + result["quality_score"] * 0.12
    ).round(2)

    result = result.sort_values(
        ["recommendation_score", "taste_match_score", "vote_average"],
        ascending=[False, False, False],
    )

    if "tmdb_id" in result.columns:
        result = result.drop_duplicates(subset=["tmdb_id"])

    # Keep a broader pre-ranked pool, then diversify it.
    result = result.head(max(limit * 4, 100)).reset_index(drop=True)

    return _diversify_ranked_candidates(
        result,
        limit=limit,
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
