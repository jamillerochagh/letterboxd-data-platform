import pandas as pd

from .config import LIKED_BOOST, MIN_RATING, WEIGHTS


# =========================================================
# USER PREFERENCES
# =========================================================

def attach_user_preferences(
    enriched,
    ratings,
    likes,
):
    result = enriched.copy()

    # -----------------------------------------------------
    # RATINGS
    # -----------------------------------------------------

    if not ratings.empty:

        rating_columns = [
            column
            for column in [
                "Letterboxd URI",
                "Rating",
            ]
            if column in ratings.columns
        ]

        rating_data = (
            ratings[
                rating_columns
            ]
            .copy()
        )

        if "Rating" in rating_data.columns:

            rating_data["Rating"] = pd.to_numeric(
                rating_data["Rating"],
                errors="coerce",
            )

            rating_data = rating_data.rename(
                columns={
                    "Rating": "rating"
                }
            )

        # Prevent duplicated ratings from creating
        # duplicated movies after the merge.

        if "Letterboxd URI" in rating_data.columns:

            rating_data = (
                rating_data
                .drop_duplicates(
                    subset=[
                        "Letterboxd URI"
                    ],
                    keep="last",
                )
            )

            result = result.merge(
                rating_data,
                on="Letterboxd URI",
                how="left",
            )

    if "rating" not in result.columns:
        result["rating"] = pd.NA

    # -----------------------------------------------------
    # LIKES
    # -----------------------------------------------------

    liked_uris = set()

    if (
        not likes.empty
        and "Letterboxd URI" in likes.columns
    ):

        liked_uris = set(
            likes[
                "Letterboxd URI"
            ]
            .dropna()
            .astype(str)
        )

    result["liked"] = (
        result[
            "Letterboxd URI"
        ]
        .astype(str)
        .isin(liked_uris)
    )

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

def build_taste_profile(df):

    data = add_decade(df)

    ratings = pd.to_numeric(
        data["rating"],
        errors="coerce",
    )

    liked = (
        data["liked"]
        .fillna(False)
        .astype(bool)
    )

    # We build the profile mainly from movies
    # the user demonstrably enjoyed.

    enjoyed = data[
        (ratings >= MIN_RATING)
        | liked
    ].copy()

    # Fallback for users without ratings/likes.

    if enjoyed.empty:
        enjoyed = data.copy()

    enjoyed["preference_weight"] = 1.0

    # -----------------------------------------------------
    # RATING WEIGHT
    # -----------------------------------------------------

    enjoyed_ratings = pd.to_numeric(
        enjoyed["rating"],
        errors="coerce",
    )

    rated_mask = (
        enjoyed_ratings.notna()
    )

    # 3.5 -> 0.70
    # 4.0 -> 0.80
    # 4.5 -> 0.90
    # 5.0 -> 1.00

    enjoyed.loc[
        rated_mask,
        "preference_weight",
    ] = (
        enjoyed_ratings[
            rated_mask
        ]
        .clip(
            lower=0.5,
            upper=5.0,
        )
        / 5.0
    )

    # Letterboxd "like" adds additional evidence.

    enjoyed.loc[
        enjoyed["liked"] == True,
        "preference_weight",
    ] *= LIKED_BOOST

    # -----------------------------------------------------
    # PROFILE BY ATTRIBUTE
    # -----------------------------------------------------

    profile = {}

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
            != ""
        ]

        if valid.empty:

            profile[attribute] = {}

            continue

        scores = (
            valid
            .groupby(attribute)[
                "preference_weight"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        # IMPORTANT:
        # Normalize relative to the strongest preference,
        # rather than dividing by the entire category.
        #
        # Strongest value = 1.0
        # Other values = relative strength.

        maximum = scores.max()

        if maximum <= 0:

            profile[attribute] = {}

            continue

        normalized_scores = (
            scores / maximum
        )

        profile[attribute] = (
            normalized_scores
            .clip(
                lower=0,
                upper=1,
            )
            .to_dict()
        )

    return profile


# =========================================================
# SCORE ONE MOVIE
# =========================================================

def score_movie(
    movie,
    profile,
):

    weighted_score = 0.0
    available_weight = 0.0

    for (
        attribute,
        attribute_weight,
    ) in WEIGHTS.items():

        # If we know nothing about this attribute
        # in the user's profile, it should NOT
        # penalize the movie.

        attribute_profile = (
            profile.get(
                attribute,
                {},
            )
        )

        if not attribute_profile:
            continue

        value = movie.get(
            attribute
        )

        if pd.isna(value):
            continue

        # Decades are stored as integers
        # in the taste profile.

        if attribute == "decade":

            try:

                value = int(value)

            except (
                TypeError,
                ValueError,
            ):

                continue

        else:

            value = str(value).strip()

            if not value:
                continue

        value_score = (
            attribute_profile.get(
                value,
                0.0,
            )
        )

        weighted_score += (
            value_score
            * attribute_weight
        )

        available_weight += (
            attribute_weight
        )

    if available_weight == 0:
        return 0.0

    # Weighted similarity from 0 to 1.

    similarity = (
        weighted_score
        / available_weight
    )

    # Convert to a readable 0–100 score.
    #
    # This is a similarity score,
    # NOT a probability.

    taste_score = (
        similarity * 100
    )

    return round(
        taste_score,
        1,
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

    result = add_decade(
        candidates
    )

    # -----------------------------------------------------
    # REMOVE WATCHED MOVIES
    # -----------------------------------------------------

    if "tmdb_id" in result.columns:

        ids = pd.to_numeric(
            result["tmdb_id"],
            errors="coerce",
        )

        result = result[
            ~ids.isin(
                watched_tmdb_ids
            )
        ].copy()

    # -----------------------------------------------------
    # TASTE SCORE
    # -----------------------------------------------------

    result[
        "taste_match_score"
    ] = result.apply(
        lambda row:
        score_movie(
            row,
            profile,
        ),
        axis=1,
    )

    result = result[
        result[
            "taste_match_score"
        ] > 0
    ].copy()

    # -----------------------------------------------------
    # FINAL RANKING
    # -----------------------------------------------------

    sort_columns = [
        "taste_match_score"
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

    result = (
        result
        .sort_values(
            sort_columns,
            ascending=ascending,
        )
    )

    if "tmdb_id" in result.columns:

        result = (
            result
            .drop_duplicates(
                subset=[
                    "tmdb_id"
                ]
            )
        )

    return (
        result
        .head(limit)
        .reset_index(
            drop=True
        )
    )


# =========================================================
# PROFILE SUMMARY
# =========================================================

def get_profile_summary(
    profile,
):

    summary = {}

    for (
        attribute,
        values,
    ) in profile.items():

        if not values:

            summary[
                attribute
            ] = None

            continue

        summary[
            attribute
        ] = max(
            values,
            key=values.get,
        )

    return summary