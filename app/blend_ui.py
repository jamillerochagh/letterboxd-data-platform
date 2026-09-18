import hashlib
import html

import pandas as pd
import streamlit as st
import traceback

from processor import (
    extract_letterboxd_zip,
    find_export_directory,
    load_letterboxd_data,
)

from enrichment import enrich_movies

from src.letterboxd_pipeline.blend import (
    blend_is_ready,
    create_blend,
    get_blend,
    get_blend_profiles,
    get_blend_recommendations,
    save_blend_profile,
    save_blend_recommendations,
)

from src.letterboxd_pipeline.blend_engine import (
    build_blend_analysis,
    rank_blend_candidates,
)

from src.letterboxd_pipeline.recommendations import (
    attach_user_preferences,
    build_taste_profile,
)

from src.letterboxd_pipeline.tmdb import (
    build_candidate_catalog,
    enrich_candidate_catalog,
    get_movie_details,
    prefilter_candidates,
)




# =========================================================
# HELPERS
# =========================================================

def _prepare_ratings(
    ratings: pd.DataFrame,
) -> pd.DataFrame:

    result = ratings.copy()

    if "Rating" in result.columns:
        result["Rating"] = pd.to_numeric(
            result["Rating"],
            errors="coerce",
        )

    return result


def _profile_history_from_upload(
    uploaded_file,
) -> pd.DataFrame:
    """
    Process one Letterboxd ZIP and return enriched movie
    history containing rating + liked information.

    The original ZIP is never persisted.
    """

    temp_dir = None

    try:
        uploaded_file.seek(0)

        temp_dir = extract_letterboxd_zip(
            uploaded_file
        )

        export_dir = find_export_directory(
            temp_dir
        )

        data = load_letterboxd_data(
            export_dir
        )

        watched = data["watched"]

        ratings = _prepare_ratings(
            data["ratings"]
        )

        likes = data["likes"]

        total_movies = len(watched)

        progress = st.progress(
            0,
            text=(
                f"Analyzing movies — "
                f"0 / {total_movies} (0%)"
            ),
        )

        def update_progress(value):

            percentage = min(
                int(value * 100),
                100,
            )

            processed = min(
                int(round(
                    value * total_movies
                )),
                total_movies,
            )

            progress.progress(
                percentage,
                text=(
                    f"Analyzing movies — "
                    f"{processed} / "
                    f"{total_movies} "
                    f"({percentage}%)"
                ),
            )

        try:

            enriched = enrich_movies(
                watched,
                progress_callback=update_progress,
            )

        finally:

            progress.empty()

        if enriched is None:
            enriched = watched.copy()

        if not isinstance(
            enriched,
            pd.DataFrame,
        ):
            enriched = pd.DataFrame(
                enriched
            )

        history = attach_user_preferences(
            enriched,
            ratings,
            likes,
        )

        return history

    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _history_from_profile(
    profile_data,
) -> pd.DataFrame:

    if not profile_data:
        return pd.DataFrame()

    history = profile_data.get(
        "movie_history",
        [],
    )

    if not history:
        return pd.DataFrame()

    return pd.DataFrame(
        history
    )


def _display_name(
    value,
    fallback,
):
    if value is None:
        return fallback

    value = str(value).strip()

    return value or fallback


# =========================================================
# MOVIE TABLE
# =========================================================

def _render_movie_table(
    movies: pd.DataFrame,
    creator_name: str,
    friend_name: str,
):

    if movies is None or movies.empty:
        st.info(
            "No movies are available for this section."
        )
        return

    columns = [
        column
        for column in [
            "title",
            "year",
            "creator_rating",
            "friend_rating",
            "rating_gap",
        ]
        if column in movies.columns
    ]

    display = movies[
        columns
    ].copy()

    display = display.rename(
        columns={
            "title": "Movie",
            "year": "Year",
            "creator_rating": creator_name.title(),
            "friend_rating": friend_name.title(),
            "rating_gap": "Rating Gap",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
    )


# =========================================================
# RECOMMENDATION CARDS
# =========================================================

def _render_blend_recommendations(
    recommendations: pd.DataFrame,
    creator_name: str,
    friend_name: str,
):

    st.subheader(
        "What Should We Watch?"
    )

    st.caption(
        "Movies neither of you has watched, ranked by "
        "how strongly they match both taste profiles."
    )

    if (
        recommendations is None
        or recommendations.empty
    ):
        st.info(
            "No shared recommendations were found."
        )
        return

    top = (
        recommendations
        .head(12)
        .copy()
    )

    for start in range(
        0,
        len(top),
        4,
    ):

        row = top.iloc[
            start:start + 4
        ]

        columns = st.columns(
            4,
            gap="medium",
        )

        for column, (_, movie) in zip(
            columns,
            row.iterrows(),
        ):

            with column:

                title = movie.get(
                    "title"
                )

                if (
                    pd.isna(title)
                    or not str(title).strip()
                ):
                    title = "Unknown Movie"

                year = movie.get(
                    "Year"
                )

                year_text = ""

                if pd.notna(year):
                    try:
                        year_text = str(
                            int(float(year))
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                genre = movie.get(
                    "genre_primary"
                )

                genre_text = (
                    str(genre)
                    if (
                        pd.notna(genre)
                        and str(genre).strip()
                    )
                    else ""
                )

                poster = movie.get(
                    "poster_path"
                )

                if (
                    pd.notna(poster)
                    and str(poster).strip()
                ):
                    st.image(
                        "https://image.tmdb.org/"
                        f"t/p/w500{poster}",
                        width="stretch",
                    )

                else:
                    st.markdown(
                        """
                        <div style="
                            width:100%;
                            aspect-ratio:2/3;
                            background:#1B2229;
                            border:1px solid #303A43;
                            border-radius:8px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            color:#A8B3BD;
                            font-size:13px;">
                            Poster unavailable
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                info = " · ".join(
                    item
                    for item in [
                        year_text,
                        genre_text,
                    ]
                    if item
                )

                blend_score = movie.get(
                    "blend_score"
                )

                creator_score = movie.get(
                    "creator_match_score"
                )

                friend_score = movie.get(
                    "friend_match_score"
                )

                st.markdown(
                    f"""
                    <div class="recommendation-title">
                        {title}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="recommendation-info">
                        {info}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if pd.notna(
                    blend_score
                ):
                    st.markdown(
                        f"""
                        <div class="recommendation-match">
                            Blend Score ·
                            {float(blend_score):.0f}/100
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if (
                    pd.notna(creator_score)
                    and pd.notna(friend_score)
                ):
                    st.caption(
                        f"{creator_name}: "
                        f"{float(creator_score):.0f} · "
                        f"{friend_name}: "
                        f"{float(friend_score):.0f}"
                    )


def _prefilter_blend_candidates(
    candidates: list[dict],
    limit: int = 90,
) -> list[dict]:
    """Reduce the discovery pool before expensive TMDB detail enrichment."""

    if not candidates:
        return []

    ranked = []

    for candidate in candidates:
        vote_average = pd.to_numeric(
            candidate.get("vote_average"),
            errors="coerce",
        )
        vote_count = pd.to_numeric(
            candidate.get("vote_count"),
            errors="coerce",
        )
        popularity = pd.to_numeric(
            candidate.get("popularity"),
            errors="coerce",
        )

        vote_average = 0.0 if pd.isna(vote_average) else float(vote_average)
        vote_count = 0.0 if pd.isna(vote_count) else float(vote_count)
        popularity = 0.0 if pd.isna(popularity) else float(popularity)

        if vote_average < 6.2 or vote_count < 150:
            continue

        discovery_score = (
            vote_average * 10
            + min(vote_count / 250, 20)
            + min(popularity / 25, 8)
        )
        ranked.append((discovery_score, candidate))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked[:limit]]


# =========================================================
# BUILD SHARED RECOMMENDATIONS
# =========================================================

@st.cache_data(
    show_spinner=False,
    ttl=60 * 60 * 24,
)
def _build_shared_recommendations(
    creator_history_records,
    friend_history_records,
):
    creator_history = pd.DataFrame(creator_history_records)
    friend_history = pd.DataFrame(friend_history_records)

    creator_profile = build_taste_profile(creator_history)
    friend_profile = build_taste_profile(friend_history)

    creator_genres = creator_profile.get(
        "genre_combined",
        creator_profile.get("genre_primary", {}),
    )
    friend_genres = friend_profile.get(
        "genre_combined",
        friend_profile.get("genre_primary", {}),
    )

    shared_genre_scores = {
        genre: min(
            float(creator_genres.get(genre, 0.0)),
            float(friend_genres.get(genre, 0.0)),
        )
        for genre in (set(creator_genres) & set(friend_genres))
    }

    preferred_genres = [
        genre
        for genre, _ in sorted(
            shared_genre_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:6]
    ]

    if not preferred_genres:
        preferred_genres = list(
            dict.fromkeys(
                list(creator_genres.keys())
                + list(friend_genres.keys())
            )
        )[:6]

    candidates = build_candidate_catalog(
        preferred_genres,
        pages_per_genre=2,
    )

    candidates = _prefilter_blend_candidates(
        candidates,
        limit=90,
    )

    if not candidates:
        return pd.DataFrame()

    candidate_df = enrich_candidate_catalog(
        candidates,
        max_workers=8,
    )

    if candidate_df.empty:
        return pd.DataFrame()

    return rank_blend_candidates(
        candidate_df,
        creator_profile,
        friend_profile,
        creator_history,
        friend_history,
        limit=30,
    )


@st.cache_data(
    show_spinner=False,
    ttl=60 * 60 * 24,
)
def _build_cached_blend_analysis(
    creator_history_records,
    friend_history_records,
):
    creator_history = pd.DataFrame(creator_history_records)
    friend_history = pd.DataFrame(friend_history_records)

    return build_blend_analysis(
        creator_history,
        friend_history,
    )


# =========================================================
# BLEND RESULTS
# =========================================================

def render_blend_result(
    blend_id: str,
):
    blend = get_blend(blend_id)

    if not blend:
        st.error("This Movie Blend does not exist or has expired.")
        return

    profiles = get_blend_profiles(blend_id)
    creator_data = profiles.get("creator")
    friend_data = profiles.get("friend")

    if not creator_data or not friend_data:
        st.info("This Blend is still waiting for both profiles.")
        return

    creator_name = _display_name(
        creator_data.get("display_name"),
        "Creator",
    ).title()

    friend_name = _display_name(
        friend_data.get("display_name"),
        "Friend",
    ).title()

    creator_history = _history_from_profile(creator_data)
    friend_history = _history_from_profile(friend_data)

    analysis = _build_cached_blend_analysis(
        creator_history.to_dict(orient="records"),
        friend_history.to_dict(orient="records"),
    )

    creator_profile = analysis["creator_profile"]
    friend_profile = analysis["friend_profile"]

    def esc(value):
        if value is None:
            return "—"
        return html.escape(str(value))

    def top_preference(profile, attribute):
        values = profile.get(attribute, {}) or {}
        return max(values, key=values.get) if values else None

    def profile_stats(history, profile):
        ratings = pd.Series(dtype=float)

        if "rating" in history.columns:
            ratings = pd.to_numeric(
                history["rating"],
                errors="coerce",
            ).dropna()

        return {
            "watched": len(history),
            "rated": len(ratings),
            "average": (
                float(ratings.mean())
                if not ratings.empty
                else None
            ),
            "genre": top_preference(profile, "genre_primary"),
            "decade": top_preference(profile, "decade"),
            "country": top_preference(profile, "country_primary"),
            "language": top_preference(
                profile,
                "original_language",
            ),
        }

    def format_decade(value):
        if value is None:
            return "—"
        try:
            return f"{int(float(value))}s"
        except (TypeError, ValueError):
            return str(value)

    def format_language(value):
        if value is None:
            return "—"

        names = {
            "en": "English",
            "fr": "French",
            "de": "German",
            "pt": "Portuguese",
            "es": "Spanish",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
        }

        return names.get(
            str(value).lower(),
            str(value),
        )

    def format_preference(attribute, value):
        if value is None:
            return "—"
        if attribute == "decade":
            return format_decade(value)
        if attribute == "original_language":
            return format_language(value)
        return str(value)

    def rating_text(value):
        if pd.isna(value):
            return "—"
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return "—"

    def poster_url(value):
        if value is None or pd.isna(value):
            return None

        value = str(value).strip()

        if not value:
            return None

        return f"https://image.tmdb.org/t/p/w342{value}"

    def strongest_shared_value(attribute):
        stored = analysis.get(
            "shared_preferences",
            {},
        ).get(attribute)

        if stored is not None:
            return stored

        creator_values = creator_profile.get(
            attribute,
            {},
        ) or {}

        friend_values = friend_profile.get(
            attribute,
            {},
        ) or {}

        common_values = set(
            creator_values
        ).intersection(friend_values)

        if not common_values:
            return None

        return max(
            common_values,
            key=lambda value: (
                float(creator_values.get(value, 0))
                + float(friend_values.get(value, 0))
            ),
        )

    creator_stats = profile_stats(
        creator_history,
        creator_profile,
    )

    friend_stats = profile_stats(
        friend_history,
        friend_profile,
    )

    both_love = analysis.get(
        "movies_both_love",
        pd.DataFrame(),
    ).copy()

    # =====================================================
    # LETTERBOXD BLEND CSS
    # =====================================================

    st.html(
        """
        <style>
        .lb-blend {
            --bg:#0B0D10;
            --surface:#151A1E;
            --surface2:#1B2229;
            --border:#303A43;
            --text:#F4F7F9;
            --muted:#A8B3BD;
            --green:#00E054;
            --blue:#40BCF4;
            --orange:#FF8000;
            color:var(--text);
            font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        }

        .lb-kicker {
            color:var(--green);
            font-size:12px;
            font-weight:800;
            letter-spacing:.16em;
            margin-bottom:6px;
        }

        .lb-title {
            color:var(--text);
            font-size:38px;
            line-height:1.05;
            font-weight:800;
            margin:0 0 8px 0;
        }

        .lb-subtitle {
            color:var(--muted);
            font-size:14px;
            margin-bottom:24px;
        }

        .lb-summary {
            display:grid;
            grid-template-columns:1.05fr 1.7fr;
            gap:14px;
            margin-bottom:34px;
        }

        .lb-panel {
            background:var(--surface);
            border:1px solid var(--border);
            border-radius:12px;
            overflow:hidden;
        }

        .lb-match {
            padding:22px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            min-height:230px;
        }

        .lb-match-label {
            color:var(--muted);
            text-transform:uppercase;
            font-size:11px;
            font-weight:800;
            letter-spacing:.12em;
        }

        .lb-match-value {
            color:var(--green);
            font-size:58px;
            line-height:1;
            font-weight:850;
            margin:8px 0 16px;
        }

        .lb-match-track {
            width:100%;
            height:8px;
            border-radius:999px;
            background:var(--surface2);
            overflow:hidden;
            margin-bottom:20px;
        }

        .lb-match-fill {
            height:100%;
            background:linear-gradient(
                90deg,
                var(--orange) 0%,
                var(--green) 50%,
                var(--blue) 100%
            );
            border-radius:999px;
        }

        .lb-kpis {
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:14px;
        }

        .lb-kpi-value {
            color:var(--text);
            font-size:23px;
            font-weight:800;
        }

        .lb-kpi-label {
            color:var(--muted);
            font-size:12px;
            margin-top:2px;
        }

        .lb-profile-head,
        .lb-profile-row {
            display:grid;
            grid-template-columns:1.2fr 1fr 1fr;
            align-items:center;
        }

        .lb-profile-head {
            background:var(--surface2);
            border-bottom:1px solid var(--border);
        }

        .lb-profile-head > div,
        .lb-profile-row > div {
            padding:13px 16px;
        }

        .lb-profile-head > div {
            font-size:13px;
            font-weight:800;
        }

        .lb-profile-head .creator {
            color:var(--orange);
        }

        .lb-profile-head .friend {
            color:var(--blue);
        }

        .lb-profile-row {
            border-bottom:1px solid rgba(48,58,67,.7);
        }

        .lb-profile-row:last-child {
            border-bottom:0;
        }

        .lb-profile-label {
            color:var(--muted);
            font-size:12px;
            text-transform:uppercase;
            letter-spacing:.05em;
        }

        .lb-profile-value {
            color:var(--text);
            font-size:14px;
            font-weight:650;
        }

        .lb-section {
            margin-top:34px;
        }

        .lb-section-title {
            color:var(--text);
            font-size:25px;
            font-weight:800;
            margin:0 0 4px 0;
        }

        .lb-section-copy {
            color:var(--muted);
            font-size:13px;
            margin-bottom:18px;
        }

        .lb-taste-row {
            display:grid;
            grid-template-columns:130px 1fr 54px;
            gap:14px;
            align-items:center;
            margin:14px 0;
        }

        .lb-taste-label {
            color:var(--text);
            font-size:13px;
            font-weight:750;
        }

        .lb-taste-value {
            color:var(--muted);
            font-size:12px;
            margin-bottom:6px;
        }

        .lb-taste-track {
            height:11px;
            background:var(--surface2);
            border:1px solid var(--border);
            border-radius:999px;
            overflow:hidden;
        }

        .lb-taste-fill {
            height:100%;
            border-radius:999px;
        }

        .lb-taste-score {
            color:var(--text);
            font-size:13px;
            font-weight:800;
            text-align:right;
        }

        .lb-carousel {
            display:flex;
            gap:14px;
            overflow-x:auto;
            padding:4px 2px 14px 2px;
            scrollbar-width:thin;
            scrollbar-color:var(--border) var(--bg);
        }

        .lb-movie-card {
            flex:0 0 160px;
        }

        .lb-poster {
            width:160px;
            aspect-ratio:2/3;
            object-fit:cover;
            border-radius:8px;
            border:1px solid var(--border);
            display:block;
            background:var(--surface2);
        }

        .lb-poster-empty {
            width:160px;
            aspect-ratio:2/3;
            border-radius:8px;
            border:1px solid var(--border);
            background:var(--surface2);
            color:var(--muted);
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            font-size:12px;
        }

        .lb-movie-title {
            color:var(--text);
            font-size:13px;
            font-weight:750;
            line-height:1.25;
            margin-top:8px;
        }

        .lb-movie-rating {
            color:var(--muted);
            font-size:11px;
            margin-top:3px;
        }

        .lb-table-wrap {
            max-width:760px;
        }

        .lb-table {
            width:100%;
            border-collapse:separate;
            border-spacing:0;
            background:var(--surface);
            border:1px solid var(--border);
            border-radius:10px;
            overflow:hidden;
        }

        .lb-table th {
            background:var(--surface2);
            color:var(--muted);
            font-size:11px;
            text-transform:uppercase;
            letter-spacing:.06em;
            text-align:left;
            padding:12px 14px;
            border-bottom:1px solid var(--border);
        }

        .lb-table th.creator {
            color:var(--orange);
        }

        .lb-table th.friend {
            color:var(--blue);
        }

        .lb-table td {
            color:var(--text);
            font-size:13px;
            padding:12px 14px;
            border-bottom:1px solid rgba(48,58,67,.7);
        }

        .lb-table tr:last-child td {
            border-bottom:0;
        }

        .lb-gap {
            color:var(--green);
            font-weight:800;
        }

        @media (max-width:800px) {
            .lb-summary {
                grid-template-columns:1fr;
            }

            .lb-profile-head,
            .lb-profile-row {
                grid-template-columns:1.2fr 1fr 1fr;
            }

            .lb-taste-row {
                grid-template-columns:95px 1fr 48px;
                gap:9px;
            }

            .lb-title {
                font-size:30px;
            }
        }
        
        .lb-kpi-help {
            color:#6F7B85;
            font-size:10px;
            line-height:1.25;
            margin-top:4px;
            max-width:220px;
            font-weight:400;
        }

        </style>
        """
    )

    # =====================================================
    # HEADER + COMPACT SUMMARY
    # =====================================================

    match = float(analysis["taste_match"])
    gap = analysis.get("average_rating_gap")
    gap_text = (
        f"{float(gap):.1f}"
        if gap is not None
        else "—"
    )

    profile_rows = [
        (
            "Films watched",
            f"{creator_stats['watched']:,}",
            f"{friend_stats['watched']:,}",
        ),
        (
            "Movies rated",
            f"{creator_stats['rated']:,}",
            f"{friend_stats['rated']:,}",
        ),
        (
            "Average rating",
            (
                f"{creator_stats['average']:.1f}"
                if creator_stats["average"] is not None
                else "—"
            ),
            (
                f"{friend_stats['average']:.1f}"
                if friend_stats["average"] is not None
                else "—"
            ),
        ),
        (
            "Favorite genre",
            creator_stats["genre"] or "—",
            friend_stats["genre"] or "—",
        ),
        (
            "Favorite decade",
            format_decade(creator_stats["decade"]),
            format_decade(friend_stats["decade"]),
        ),
        (
            "Top country",
            creator_stats["country"] or "—",
            friend_stats["country"] or "—",
        ),
        (
            "Top language",
            format_language(creator_stats["language"]),
            format_language(friend_stats["language"]),
        ),
    ]

    profile_html = "".join(
        (
            '<div class="lb-profile-row">'
            f'<div class="lb-profile-label">{esc(label)}</div>'
            f'<div class="lb-profile-value">{esc(left)}</div>'
            f'<div class="lb-profile-value">{esc(right)}</div>'
            "</div>"
        )
        for label, left, right in profile_rows
    )

    st.html(
        f"""
        <div class="lb-blend">
            <div class="lb-kicker">MOVIE BLEND</div>
            <div class="lb-title">
                {esc(creator_name)} × {esc(friend_name)}
            </div>
            <div class="lb-subtitle">
                See where your movie tastes align, where they clash,
                and what you should watch together.
            </div>

            <div class="lb-summary">
                <div class="lb-panel lb-match">
                    <div class="lb-match-label">Taste Match</div>
                    <div class="lb-match-value">{match:.0f}%</div>

                    <div class="lb-match-track">
                        <div
                            class="lb-match-fill"
                            style="width:{max(0, min(match, 100)):.1f}%">
                        </div>
                    </div>

                    <div class="lb-kpis">
                        <div>
                            <div class="lb-kpi-value">
                                {analysis['movies_in_common']:,}
                            </div>
                            <div class="lb-kpi-label">
                                movies in common
                            </div>
                        </div>

                        <div>
                            <div class="lb-kpi-value">
                                {gap_text}
                            </div>
                            <div class="lb-kpi-label">
                                average rating gap
                            </div>
                            <div class="lb-kpi-help">Average difference between your ratings for movies you both rated.</div>
                        </div>
                    </div>
                </div>

                <div class="lb-panel">
                    <div class="lb-profile-head">
                        <div>Profile</div>
                        <div class="creator">{esc(creator_name)}</div>
                        <div class="friend">{esc(friend_name)}</div>
                    </div>
                    {profile_html}
                </div>
            </div>
        </div>
        """
    )

    # =====================================================
    # WHAT YOU HAVE IN COMMON
    # =====================================================

    attribute_matches = analysis.get(
        "attribute_matches",
        {},
    )

    config = [
        ("genre_primary", "Genre", "#00E054"),
        ("genre_secondary", "Secondary genre", "#40BCF4"),
        ("decade", "Decade", "#FF8000"),
        ("country_primary", "Country", "#00E054"),
        ("original_language", "Language", "#40BCF4"),
        ("director", "Director", "#FF8000"),
    ]

    shared_rows = []

    for attribute, label, color in config:
        if attribute not in attribute_matches:
            continue

        score = float(
            attribute_matches[attribute]
        )

        shared_value = strongest_shared_value(
            attribute
        )

        shared_rows.append(
            (
                attribute,
                label,
                color,
                score,
                format_preference(
                    attribute,
                    shared_value,
                ),
            )
        )

    shared_rows.sort(
        key=lambda item: item[3],
        reverse=True,
    )

    taste_html = ""

    for _, label, color, score, value in shared_rows:
        taste_html += f"""
        <div class="lb-taste-row">
            <div class="lb-taste-label">
                {esc(label)}
            </div>

            <div>
                <div class="lb-taste-value">
                    {esc(value)}
                </div>

                <div class="lb-taste-track">
                    <div
                        class="lb-taste-fill"
                        style="
                            width:{max(0, min(score, 100)):.1f}%;
                            background:{color};
                        ">
                    </div>
                </div>
            </div>

            <div class="lb-taste-score">
                {score:.0f}%
            </div>
        </div>
        """

    if taste_html:
        st.html(
            f"""
            <div class="lb-blend lb-section">
                <div class="lb-section-title">
                    What You Have in Common
                </div>
                <div class="lb-section-copy">
                    Your strongest shared preferences across
                    genres, eras, countries, languages and filmmakers.
                </div>

                <div class="lb-panel" style="padding:18px 20px;">
                    {taste_html}
                </div>
            </div>
            """
        )
    else:
        st.subheader("What You Have in Common")
        st.info("Not enough shared taste data yet.")

    # =====================================================
    # YOU BOTH LOVE
    # =====================================================

    if both_love.empty:
        st.html(
            """
            <div class="lb-blend lb-section">
                <div class="lb-section-title">You Both Love</div>
                <div class="lb-section-copy">
                    Movies both of you rated exactly 5.0.
                </div>
            </div>
            """
        )
        st.info(
            "You do not have a shared 5-star movie yet."
        )
    else:
        cards = ""

        for _, movie in both_love.iterrows():
            title = movie.get(
                "title",
                "Unknown movie",
            )

            image = poster_url(
                movie.get("poster_path")
            )

            if image:
                visual = (
                    f'<img class="lb-poster" '
                    f'src="{esc(image)}" '
                    f'alt="{esc(title)} poster">'
                )
            else:
                visual = (
                    '<div class="lb-poster-empty">'
                    "Poster unavailable"
                    "</div>"
                )

            cards += f"""
            <div class="lb-movie-card">
                {visual}
                <div class="lb-movie-title">
                    {esc(title)}
                </div>
                <div class="lb-movie-rating">
                    {esc(creator_name)} {esc(rating_text(movie.get("creator_rating")))}
                    ·
                    {esc(friend_name)} {esc(rating_text(movie.get("friend_rating")))}
                </div>
            </div>
            """

        st.html(
            f"""
            <div class="lb-blend lb-section">
                <div class="lb-section-title">
                    You Both Love
                </div>
                <div class="lb-section-copy">
                    Movies both of you rated exactly 5.0.
                </div>

                <div class="lb-carousel">
                    {cards}
                </div>
            </div>
            """
        )

    # =====================================================
    # BIGGEST DISAGREEMENTS
    # =====================================================

    disagreements = analysis.get(
        "biggest_disagreements",
        pd.DataFrame(),
    ).copy()

    if disagreements.empty:
        st.html(
            """
            <div class="lb-blend lb-section">
                <div class="lb-section-title">
                    Biggest Disagreements
                </div>
            </div>
            """
        )
        st.info(
            "No major rating disagreements yet."
        )
    else:
        rows_html = ""

        for _, movie in disagreements.iterrows():
            gap = movie.get("rating_gap")

            rows_html += f"""
            <tr>
                <td>
                    {esc(movie.get("title", "Unknown movie"))}
                </td>
                <td>
                    {esc(rating_text(movie.get("creator_rating")))}
                </td>
                <td>
                    {esc(rating_text(movie.get("friend_rating")))}
                </td>
                <td>
                    <span class="lb-gap">
                        {
                            f"{float(gap):.1f}"
                            if pd.notna(gap)
                            else "—"
                        }
                    </span>
                </td>
            </tr>
            """

        st.html(
            f"""
            <div class="lb-blend lb-section">
                <div class="lb-section-title">
                    Biggest Disagreements
                </div>
                <div class="lb-section-copy">
                    The movies where your ratings differ the most.
                </div>

                <div class="lb-table-wrap">
                <table class="lb-table">
                    <thead>
                        <tr>
                            <th>Movie</th>
                            <th class="creator">
                                {esc(creator_name)}
                            </th>
                            <th class="friend">
                                {esc(friend_name)}
                            </th>
                            <th>Difference</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                </div>
            </div>
            """
        )

    # =====================================================
    # WHAT SHOULD WE WATCH
    # =====================================================

    st.divider()

    # The visual Blend above is intentionally rendered before this expensive
    # section. Users can read their results while recommendations are built.
    persisted_recommendations = get_blend_recommendations(
        blend_id
    )

    if persisted_recommendations is not None:
        _render_blend_recommendations(
            persisted_recommendations,
            creator_name,
            friend_name,
        )
    else:
        recommendation_status = st.status(
            "Finding movies for both of you...",
            expanded=True,
        )

        try:
            recommendation_status.write(
                "Your Blend results are ready."
            )
            recommendation_status.write(
                "Now building shared recommendations..."
            )

            recommendations = (
                _build_shared_recommendations(
                    creator_history.to_dict(
                        orient="records"
                    ),
                    friend_history.to_dict(
                        orient="records"
                    ),
                )
            )

            # Persist the expensive derived result. If the database write
            # fails, the current page still works with the in-memory result.
            save_blend_recommendations(
                blend_id,
                recommendations,
            )

            recommendation_status.update(
                label="Shared recommendations ready",
                state="complete",
                expanded=False,
            )

            _render_blend_recommendations(
                recommendations,
                creator_name,
                friend_name,
            )

        except Exception as error:
            recommendation_status.update(
                label="Shared recommendations unavailable",
                state="error",
                expanded=False,
            )

            print(
                "BLEND RECOMMENDATION ERROR | "
                f"{type(error).__name__}: {error}"
            )

            st.info(
                "Your Blend results are ready, but shared "
                "recommendations could not be generated."
            )


# =========================================================
# CREATOR
# =========================================================

def render_create_blend(
    enriched: pd.DataFrame,
    ratings: pd.DataFrame,
    likes: pd.DataFrame,
):
    """Creator flow with explicit, deterministic UI state."""
    st.header(
        "Movie Blend"
    )

    st.caption(
        "Compare your movie taste with a friend "
        "and discover what you should watch together."
    )

    created_blend_id = st.session_state.get(
        "created_blend_id"
    )

    if created_blend_id:
        blend = get_blend(
            created_blend_id
        )

        if not blend:
            st.session_state.pop(
                "created_blend_id",
                None,
            )
            st.session_state[
                "blend_creator_state"
            ] = "idle"
            st.rerun()

        creator_name = (
            blend.get("creator_name")
            or "Your friend"
        )

        ready = bool(
            blend.get("creator_ready")
            and blend.get("friend_ready")
        )

        if ready:
            st.success(
                "Your friend is ready. Loading your Movie Blend..."
            )
            render_blend_result(
                created_blend_id
            )
            return

        st.success(
            "Your profile is ready."
        )

        st.subheader(
            "Waiting for your friend..."
        )

        st.caption(
            "Send this invitation link to your friend. "
            "The result will unlock when they finish their profile."
        )

        base_url = str(
            st.context.url
        ).rstrip("/")

        blend_url = (
            f"{base_url}/?blend={created_blend_id}"
        )

        st.text_input(
            "Invitation link",
            value=blend_url,
            key=f"blend_url_{created_blend_id}",
        )

        check_message_key = (
            f"blend_check_message_{created_blend_id}"
        )

        if st.button(
            "Check if my friend is ready",
            key=f"blend_check_{created_blend_id}",
            type="primary",
            width="stretch",
        ):
            st.session_state[
                f"_blend_check_just_clicked_{created_blend_id}"
            ] = True

            # Render feedback in the same run; persist the final message.
            check_placeholder = st.empty()
            check_placeholder.info(
                "Checking your Blend..."
            )

            try:
                refreshed_blend = get_blend(
                    created_blend_id
                )

                ready = bool(
                    refreshed_blend
                    and refreshed_blend.get(
                        "creator_ready"
                    )
                    and refreshed_blend.get(
                        "friend_ready"
                    )
                )

                if ready:
                    st.session_state.pop(
                        check_message_key,
                        None,
                    )
                    check_placeholder.success(
                        "Your friend is ready. Loading your Movie Blend..."
                    )
                    st.rerun()
                else:
                    message = (
                        "Your friend hasn't finished uploading "
                        "their profile yet."
                    )
                    st.session_state[
                        check_message_key
                    ] = message
                    check_placeholder.info(
                        message
                    )

            except Exception as error:
                message = (
                    "Unable to check the Blend right now. "
                    "Try again in a moment."
                )
                st.session_state[
                    check_message_key
                ] = message
                check_placeholder.error(
                    message
                )
                print(
                    "BLEND STATUS ERROR | "
                    f"{type(error).__name__}: {error}"
                )

        previous_message = st.session_state.get(
            check_message_key
        )

        # On the click run, feedback is already displayed in the placeholder.
        # On later reruns, keep the last status visible without duplicating it.
        if previous_message and not st.session_state.get(
            f"_blend_check_just_clicked_{created_blend_id}",
            False,
        ):
            st.info(
                previous_message
            )

        st.session_state.pop(
            f"_blend_check_just_clicked_{created_blend_id}",
            None,
        )

        st.caption(
            "The original Letterboxd ZIP files are not stored."
        )
        return

    st.markdown(
        "### Create your Movie Blend"
    )

    creator_name = st.text_input(
        "Your name",
        placeholder="Enter your name",
        key="blend_creator_name",
    )

    create_clicked = st.button(
        "Create Movie Blend",
        type="primary",
        width="stretch",
        key="create_movie_blend_button",
    )

    if not create_clicked:
        return

    if not creator_name.strip():
        st.error(
            "Enter your name before creating the Blend."
        )
        return

    # Guard against duplicate DB inserts from rapid/repeated clicks.
    if st.session_state.get(
        "blend_creator_state"
    ) == "creating":
        st.info(
            "Your Movie Blend is already being created..."
        )
        return

    st.session_state[
        "blend_creator_state"
    ] = "creating"

    create_status = st.status(
        "Creating your Movie Blend...",
        expanded=True,
    )

    try:
        create_status.write(
            "Preparing your movie taste profile..."
        )

        creator_history = (
            attach_user_preferences(
                enriched,
                ratings,
                likes,
            )
        )

        creator_profile = (
            build_taste_profile(
                creator_history
            )
        )

        create_status.write(
            "Creating your private Blend link..."
        )

        blend_id = create_blend(
            creator_name.strip()
        )

        create_status.write(
            "Saving your profile..."
        )

        save_blend_profile(
            blend_id=blend_id,
            profile_slot="creator",
            display_name=creator_name.strip(),
            taste_profile=creator_profile,
            movie_history=creator_history,
        )

        st.session_state[
            "created_blend_id"
        ] = blend_id

        st.session_state[
            "blend_creator_state"
        ] = "waiting"

        create_status.update(
            label="Movie Blend created",
            state="complete",
            expanded=False,
        )

        # One intentional rerun AFTER the completed id has been persisted.
        # The next run enters the waiting branch above; no second click needed.
        st.rerun()

    except Exception as error:
        st.session_state[
            "blend_creator_state"
        ] = "idle"

        create_status.update(
            label="Unable to create the Movie Blend",
            state="error",
            expanded=False,
        )

        st.error(
            "Unable to create the Movie Blend."
        )

        print(
            "CREATE BLEND ERROR | "
            f"{type(error).__name__}: {error}"
        )
        traceback.print_exc()


def render_blend_invitation(
    blend_id: str,
):

    with st.spinner(
        "Loading Movie Blend..."
    ):
        blend = get_blend(
            blend_id
        )

    if not blend:

        st.error(
            "This Movie Blend does not exist or has expired."
        )
        return

    ready = bool(
        blend.get("creator_ready")
        and blend.get("friend_ready")
    )

    if ready:
        st.info(
            "Both profiles are ready. Loading your Movie Blend..."
        )
        render_blend_result(
            blend_id
        )
        return

    creator_name = _display_name(
        blend.get(
            "creator_name"
        ),
        "A friend",
    )

    st.title(
        "Movie Blend"
    )

    st.subheader(
        f"{creator_name} invited you "
        "to compare your movie taste."
    )

    st.caption(
        "Upload your Letterboxd export to discover "
        "what you have in common and what you "
        "should watch together."
    )

    friend_name = st.text_input(
        "Your name",
        key=f"friend_name_{blend_id}",
    )

    uploaded_file = st.file_uploader(
        "Upload your Letterboxd export (.zip)",
        type=["zip"],
        key=f"friend_zip_{blend_id}",
    )

    if st.button(
        "Create our Movie Blend",
        type="primary",
        key=f"friend_submit_{blend_id}",
    ):

        if not friend_name.strip():

            st.warning(
                "Enter your name first."
            )

            return

        if uploaded_file is None:

            st.warning(
                "Upload your Letterboxd export first."
            )

            return

        try:
            blend_status = st.status(
                "Preparing your Movie Blend...",
                expanded=True,
            )

            upload_bytes = (
                uploaded_file.getvalue()
            )

            fingerprint = (
                hashlib.sha256(
                    upload_bytes
                ).hexdigest()
            )

            processing_key = (
                "blend_friend_fingerprint"
            )

            previous = (
                st.session_state.get(
                    processing_key
                )
            )

            if (
                previous == fingerprint
                and st.session_state.get(
                    "blend_friend_history"
                )
                is not None
            ):
                blend_status.write(
                    "Using your already processed movie history."
                )
                friend_history = (
                    st.session_state[
                        "blend_friend_history"
                    ]
                )

            else:
                blend_status.write(
                    "Analyzing and enriching your movie history..."
                )

                friend_history = (
                    _profile_history_from_upload(
                        uploaded_file
                    )
                )

                st.session_state[
                    processing_key
                ] = fingerprint

                st.session_state[
                    "blend_friend_history"
                ] = friend_history

            blend_status.write(
                "Movie analysis complete. Building your taste profile..."
            )

            friend_profile = (
                build_taste_profile(
                    friend_history
                )
            )

            blend_status.write(
                "Saving your profile and connecting both sides..."
            )

            save_blend_profile(
                blend_id=blend_id,
                profile_slot="friend",
                display_name=friend_name.strip(),
                taste_profile=friend_profile,
                movie_history=friend_history,
            )

            blend_status.update(
                label="Profiles connected. Opening your results...",
                state="complete",
                expanded=False,
            )

            # Do not claim that recommendations are already finished.
            # The next render shows the core Blend first, then builds/persists
            # recommendations underneath it.
            st.rerun()

        except Exception as error:

            print(
                "FRIEND BLEND ERROR | "
                f"{type(error).__name__}: "
                f"{error}"
            )

            st.error(
                f"Unable to process this export: {error}"
            )