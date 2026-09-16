import hashlib
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from processor import (
    extract_letterboxd_zip,
    find_export_directory,
    load_letterboxd_data,
)

from enrichment import enrich_movies

from src.letterboxd_pipeline.recommendations import (
    attach_user_preferences,
    build_taste_profile,
    get_profile_summary,
    rank_candidates,
)

from src.letterboxd_pipeline.tmdb import (
    build_candidate_catalog,
    enrich_candidate_catalog,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Letterboxd Analytics",
    layout="wide",
)


# =========================================================
# LETTERBOXD THEME
# =========================================================

BACKGROUND = "#0B0D10"
CARD = "#151A1E"
BORDER = "#303A43"

TEXT = "#F4F7F9"
MUTED = "#A8B3BD"

GREEN = "#00E054"
ORANGE = "#FF8000"
BLUE = "#40BCF4"


st.markdown(
    """
    <style>
    :root {
        --lb-bg:#0B0D10; --lb-surface:#151A1E; --lb-surface-2:#1B2229;
        --lb-border:#303A43; --lb-text:#F4F7F9; --lb-muted:#A8B3BD;
        --lb-green:#00E054; --lb-orange:#FF8000; --lb-blue:#40BCF4;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:var(--lb-bg)!important; color:var(--lb-text)!important;
    }
    [data-testid="stHeader"] { background:rgba(11,13,16,.96)!important; }
    [data-testid="stMainBlockContainer"] { padding-top:2rem; }
    h1,h2,h3,h4,h5,h6,[data-testid="stMarkdownContainer"] { color:var(--lb-text); }
    .stCaption,[data-testid="stCaptionContainer"] { color:var(--lb-muted)!important; }

    [data-testid="stMetric"] {
        background:var(--lb-surface)!important; border:1px solid var(--lb-border)!important;
        border-radius:12px!important; padding:18px!important;
        box-shadow:0 8px 24px rgba(0,0,0,.16);
    }
    [data-testid="stMetricLabel"] { color:var(--lb-muted)!important; }
    [data-testid="stMetricValue"] { color:var(--lb-text)!important; }

    [data-testid="stFileUploader"] {
        background:var(--lb-surface)!important; border:1px solid var(--lb-border)!important;
        border-radius:12px!important; padding:18px!important; color:var(--lb-text)!important;
    }
    [data-testid="stFileUploaderDropzone"],[data-testid="stFileUploader"] section {
        background:var(--lb-surface-2)!important; border:1px dashed #56636F!important;
        border-radius:10px!important; color:var(--lb-text)!important;
    }
    [data-testid="stFileUploader"] section *,[data-testid="stFileUploaderFile"] * {
        color:var(--lb-text)!important;
    }
    [data-testid="stFileUploader"] section small { color:var(--lb-muted)!important; }
    [data-testid="stFileUploader"] button {
        background:#27313A!important; color:var(--lb-text)!important;
        border:1px solid #56636F!important;
    }
    [data-testid="stFileUploaderFile"] {
        background:var(--lb-surface-2)!important; border:1px solid var(--lb-border)!important;
        color:var(--lb-text)!important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap:6px; background:transparent!important; border-bottom:1px solid var(--lb-border);
    }
    .stTabs [data-baseweb="tab"] {
        background:transparent!important; color:var(--lb-muted)!important;
        border-radius:0!important; padding:10px 16px!important;
        border-bottom:3px solid transparent!important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color:var(--lb-text)!important; border-bottom-color:var(--lb-green)!important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color:var(--lb-green)!important; }

    [data-testid="stAlert"] {
        background:var(--lb-surface-2)!important; color:var(--lb-text)!important;
        border:1px solid var(--lb-border)!important; border-radius:10px!important;
    }
    [data-testid="stDataFrame"],[data-testid="stTable"] {
        border:1px solid var(--lb-border); border-radius:10px; overflow:hidden;
    }

    input { color:var(--lb-text)!important; }
    input::placeholder { color:var(--lb-muted)!important; opacity:1; }
    [data-baseweb="input"]>div,[data-baseweb="select"]>div {
        background:var(--lb-surface-2)!important; border-color:var(--lb-border)!important;
        color:var(--lb-text)!important;
    }
    hr { border-color:var(--lb-border)!important; }

    .taste-card,.crowd-card {
        background:var(--lb-surface)!important; border-color:var(--lb-border)!important;
        box-shadow:0 8px 24px rgba(0,0,0,.14);
    }
    .recommendation-title {
        color:var(--lb-text); font-size:16px; font-weight:650; line-height:1.25; margin-top:8px;
    }
    .recommendation-info,.recommendation-tmdb { color:var(--lb-muted); font-size:13px; }
    .recommendation-match {
        color:var(--lb-green); font-size:13px; font-weight:650; margin-top:4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CHART STYLE
# =========================================================

def style_chart(
    figure,
    show_legend=False,
):

    figure.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font={
            "color": TEXT,
        },
        showlegend=show_legend,
        margin={
            "l": 20,
            "r": 35,
            "t": 30,
            "b": 20,
        },
        hoverlabel={
            "bgcolor": CARD,
            "font_color": TEXT,
        },
    )

    figure.update_xaxes(
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont={
            "color": MUTED,
        },
        title_font={
            "color": MUTED,
        },
    )

    figure.update_yaxes(
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont={
            "color": MUTED,
        },
        title_font={
            "color": MUTED,
        },
    )

    return figure


# =========================================================
# DATA HELPERS
# =========================================================

def prepare_ratings(
    ratings: pd.DataFrame,
) -> pd.DataFrame:

    result = ratings.copy()

    if "Rating" in result.columns:
        result["Rating"] = pd.to_numeric(
            result["Rating"],
            errors="coerce",
        )

    return result


def prepare_diary(
    diary: pd.DataFrame,
) -> pd.DataFrame:

    result = diary.copy()

    if "Watched Date" in result.columns:
        result["Watched Date"] = pd.to_datetime(
            result["Watched Date"],
            errors="coerce",
        )

    return result


def calculate_rewatches(
    diary: pd.DataFrame,
) -> int:

    if "Rewatch" not in diary.columns:
        return 0

    return int(
        diary["Rewatch"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "yes",
                "true",
            ]
        )
        .sum()
    )


def calculate_hours_watched(
    enriched: pd.DataFrame,
    diary: pd.DataFrame,
) -> float:
    """
    Estimate total viewing hours.

    Each movie in watched.csv counts once.
    Additional diary rewatches count as additional sessions.
    """

    if enriched.empty:
        return 0.0

    movies = enriched.copy()

    movies["runtime_numeric"] = pd.to_numeric(
        movies.get(
            "runtime_min",
            pd.Series(
                index=movies.index,
                dtype=float,
            ),
        ),
        errors="coerce",
    )

    # Every unique watched movie represents at least one viewing.

    base_minutes = (
        movies["runtime_numeric"]
        .fillna(0)
        .sum()
    )

    if (
        diary.empty
        or "Rewatch" not in diary.columns
        or "Letterboxd URI" not in diary.columns
        or "Letterboxd URI" not in movies.columns
    ):
        return base_minutes / 60

    rewatches = diary.copy()

    rewatch_mask = (
        rewatches["Rewatch"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "yes",
                "true",
            ]
        )
    )

    rewatches = rewatches[
        rewatch_mask
    ].copy()

    if rewatches.empty:
        return base_minutes / 60

    runtime_lookup = (
        movies[
            [
                "Letterboxd URI",
                "runtime_numeric",
            ]
        ]
        .drop_duplicates(
            subset=[
                "Letterboxd URI"
            ]
        )
    )

    rewatches = rewatches.merge(
        runtime_lookup,
        on="Letterboxd URI",
        how="left",
    )

    rewatch_minutes = (
        rewatches[
            "runtime_numeric"
        ]
        .fillna(0)
        .sum()
    )

    total_minutes = (
        base_minutes
        + rewatch_minutes
    )

    return total_minutes / 60


# =========================================================
# AUTOMATIC ENRICHMENT
# =========================================================

def get_enriched_movies(
    watched: pd.DataFrame,
    upload_fingerprint: str,
) -> pd.DataFrame:

    previous_fingerprint = st.session_state.get(
        "upload_fingerprint"
    )

    if previous_fingerprint != upload_fingerprint:
        st.session_state.pop("enriched_movies", None)
        st.session_state.pop("recommendations", None)
        st.session_state["upload_fingerprint"] = upload_fingerprint

    if "enriched_movies" not in st.session_state:

        with st.spinner(
            "Analyzing your movie history..."
        ):

            enriched = enrich_movies(
                watched,
            )

            if enriched is None:
                enriched = watched.copy()

            if not isinstance(enriched, pd.DataFrame):
                enriched = pd.DataFrame(enriched)

            expected_columns = {
                "tmdb_id": pd.NA,
                "director": pd.NA,
                "genre_primary": pd.NA,
                "genre_secondary": pd.NA,
                "genre_tertiary": pd.NA,
                "country_primary": pd.NA,
                "original_language": pd.NA,
                "runtime_min": pd.NA,
                "vote_average": pd.NA,
                "popularity": pd.NA,
                "tagline": pd.NA,
                "overview": pd.NA,
                "poster_path": pd.NA,
            }

            for column, default_value in expected_columns.items():
                if column not in enriched.columns:
                    enriched[column] = default_value

            for column in [
                "tmdb_id",
                "runtime_min",
                "vote_average",
                "popularity",
            ]:
                enriched[column] = pd.to_numeric(
                    enriched[column],
                    errors="coerce",
                )

            st.session_state["enriched_movies"] = enriched

    return st.session_state["enriched_movies"]


# =========================================================
# AUTOMATIC RECOMMENDATIONS
# =========================================================

def get_recommendations(
    enriched: pd.DataFrame,
    ratings: pd.DataFrame,
    likes: pd.DataFrame,
):

    preference_data = (
        attach_user_preferences(
            enriched,
            ratings,
            likes,
        )
    )

    profile = build_taste_profile(
        preference_data
    )

    if (
        "recommendations"
        not in st.session_state
    ):

        with st.spinner(
            "Building personalized recommendations..."
        ):

            preferred_genres = list(
                profile.get(
                    "genre_primary",
                    {},
                ).keys()
            )

            candidates = (
                build_candidate_catalog(
                    preferred_genres,
                    pages_per_genre=2,
                )
            )

            candidate_df = (
                enrich_candidate_catalog(
                    candidates
                )
            )

            watched_ids = set(
                pd.to_numeric(
                    enriched.get(
                        "tmdb_id",
                        pd.Series(
                            dtype=float
                        ),
                    ),
                    errors="coerce",
                )
                .dropna()
                .astype(int)
            )

            recommendations = (
                rank_candidates(
                    candidate_df,
                    profile,
                    watched_ids,
                    limit=50,
                )
            )

            st.session_state[
                "recommendations"
            ] = recommendations

    return (
        profile,
        st.session_state[
            "recommendations"
        ],
    )


# =========================================================
# OVERVIEW
# =========================================================

def render_overview(
    watched,
    ratings,
    diary,
    reviews,
    likes,
    enriched,
):

    st.header(
        "Overview"
    )

    # -----------------------------------------------------
    # KPIs
    # -----------------------------------------------------

    valid_ratings = pd.to_numeric(
        ratings.get(
            "Rating",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    hours_watched = (
        calculate_hours_watched(
            enriched,
            diary,
        )
    )

    rewatches = calculate_rewatches(
        diary
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Movies watched",
        f"{len(watched):,}",
    )

    col2.metric(
        "Hours watched",
        f"{hours_watched:,.0f}",
    )

    col3.metric(
        "Average rating",
        (
            f"{valid_ratings.mean():.2f}"
            if valid_ratings.notna().any()
            else "N/A"
        ),
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Liked movies",
        f"{len(likes):,}",
    )

    col5.metric(
        "Reviews",
        f"{len(reviews):,}",
    )

    col6.metric(
        "Rewatches",
        f"{rewatches:,}",
    )

    st.divider()

    # -----------------------------------------------------
    # VIEWING ACTIVITY
    # -----------------------------------------------------

    st.subheader(
        "Viewing Activity"
    )

    if (
        diary.empty
        or "Watched Date"
        not in diary.columns
    ):

        st.info(
            "No viewing activity available."
        )

        return

    valid_diary = (
        diary
        .dropna(
            subset=[
                "Watched Date"
            ]
        )
        .copy()
    )

    if valid_diary.empty:

        st.info(
            "No dated diary entries available."
        )

        return

    monthly = (
        valid_diary
        .set_index(
            "Watched Date"
        )
        .resample("MS")
        .size()
        .reset_index(
            name="Movies"
        )
    )

    activity_chart = px.line(
        monthly,
        x="Watched Date",
        y="Movies",
        markers=True,
        color_discrete_sequence=[
            BLUE
        ],
    )

    label_interval = max(
        1,
        len(monthly) // 18,
    )

    labels = [
        str(value)
        if index % label_interval == 0
        else ""
        for index, value in enumerate(
            monthly["Movies"]
        )
    ]

    activity_chart.update_traces(
        text=labels,
        mode="lines+markers+text",
        textposition="top center",
        line={
            "width": 3,
        },
        marker={
            "size": 7,
        },
        textfont={
            "color": TEXT,
        },
    )

    activity_chart.update_layout(
        xaxis_title=None,
        yaxis_title="Movies",
    )

    activity_chart = style_chart(
        activity_chart
    )

    st.plotly_chart(
        activity_chart,
        width="stretch",
    )

    # -----------------------------------------------------
    # YEAR + MONTH
    # -----------------------------------------------------

    left, right = st.columns(2)

    # -----------------------------------------------------
    # MOVIES BY YEAR
    # -----------------------------------------------------

    with left:

        st.subheader(
            "Movies by Year"
        )

        yearly = (
            valid_diary[
                "Watched Date"
            ]
            .dt.year
            .value_counts()
            .sort_index()
            .rename_axis(
                "Year"
            )
            .reset_index(
                name="Movies"
            )
        )

        year_chart = px.bar(
            yearly,
            x="Year",
            y="Movies",
            text="Movies",
            color_discrete_sequence=[
                GREEN
            ],
        )

        year_chart.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        year_chart.update_layout(
            xaxis_title=None,
            yaxis_title="Movies",
        )

        year_chart = style_chart(
            year_chart
        )

        st.plotly_chart(
            year_chart,
            width="stretch",
        )

    # -----------------------------------------------------
    # MOVIES BY MONTH
    # -----------------------------------------------------

    with right:

        st.subheader(
            "Movies by Month"
        )

        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }

        monthly_totals = (
            valid_diary[
                "Watched Date"
            ]
            .dt.month
            .value_counts()
            .reindex(
                range(1, 13),
                fill_value=0,
            )
            .rename_axis(
                "Month number"
            )
            .reset_index(
                name="Movies"
            )
        )

        monthly_totals[
            "Month"
        ] = (
            monthly_totals[
                "Month number"
            ]
            .map(
                month_names
            )
        )

        month_chart = px.bar(
            monthly_totals,
            x="Month",
            y="Movies",
            text="Movies",
            color_discrete_sequence=[
                ORANGE
            ],
        )

        month_chart.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        month_chart.update_layout(
            xaxis_title=None,
            yaxis_title="Movies",
        )

        month_chart = style_chart(
            month_chart
        )

        st.plotly_chart(
            month_chart,
            width="stretch",
        )

    # -----------------------------------------------------
    # RECENT ACTIVITY
    # -----------------------------------------------------

    st.subheader(
        "Recent Activity"
    )

    recent = (
        valid_diary
        .sort_values(
            "Watched Date",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    recent_columns = [
        column
        for column in [
            "Watched Date",
            "Name",
            "Year",
            "Rating",
            "Rewatch",
        ]
        if column in recent.columns
    ]

    if recent_columns:

        st.dataframe(
            recent[
                recent_columns
            ],
            width="stretch",
            hide_index=True,
        )


# =========================================================
# YOUR TASTE
# =========================================================

def render_your_taste(
    enriched,
    ratings,
    likes,
    profile,
):

    st.header("Your Taste")

    # =====================================================
    # PREPARE DATA
    # =====================================================

    data = attach_user_preferences(
        enriched,
        ratings,
        likes,
    )

    if "vote_average" not in data.columns:
        data["vote_average"] = pd.NA

    data["vote_average"] = pd.to_numeric(
        data["vote_average"],
        errors="coerce",
    )

    data["runtime_numeric"] = pd.to_numeric(
        data.get(
            "runtime_min",
            pd.Series(index=data.index, dtype=float),
        ),
        errors="coerce",
    )

    data["rating_numeric"] = pd.to_numeric(
        data.get(
            "rating",
            pd.Series(index=data.index, dtype=float),
        ),
        errors="coerce",
    )

    data["year_numeric"] = pd.to_numeric(
        data.get(
            "Year",
            pd.Series(index=data.index, dtype=float),
        ),
        errors="coerce",
    )

    data["decade"] = (
        (data["year_numeric"] // 10) * 10
    ).astype("Int64")

    # =====================================================
    # KPI VALUES
    # =====================================================

    top_genre = "N/A"

    if "genre_primary" in data.columns:
        genres = (
            data["genre_primary"]
            .replace("", pd.NA)
            .dropna()
        )

        if not genres.empty:
            top_genre = genres.value_counts().index[0]

    most_watched_director = "N/A"

    if "director" in data.columns:
        directors = (
            data["director"]
            .replace("", pd.NA)
            .dropna()
        )

        if not directors.empty:
            most_watched_director = (
                directors.value_counts().index[0]
            )

    valid_decades = data["decade"].dropna()

    favorite_decade = "N/A"

    if not valid_decades.empty:
        decade_value = (
            valid_decades
            .value_counts()
            .index[0]
        )

        favorite_decade = f"{int(decade_value)}s"

    average_runtime = (
        data["runtime_numeric"]
        .dropna()
        .mean()
    )

    average_runtime_label = (
        f"{average_runtime:.0f} min"
        if pd.notna(average_runtime)
        else "N/A"
    )

    # =====================================================
    # KPI CARDS
    # =====================================================

    st.markdown(
        """
        <style>

        .taste-card {
            background-color: #151A1E;
            border: 1px solid #303A43;
            border-radius: 10px;
            padding: 18px;
            min-height: 115px;
        }

        .taste-card-label {
            color: #A8B3BD;
            font-size: 14px;
            margin-bottom: 8px;
        }

        .taste-card-value {
            color: #FFFFFF;
            font-size: 24px;
            font-weight: 600;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="taste-card">
                <div class="taste-card-label">
                    Top Genre
                </div>
                <div class="taste-card-value">
                    {top_genre}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="taste-card">
                <div class="taste-card-label">
                    Most Watched Director
                </div>
                <div class="taste-card-value">
                    {most_watched_director}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="taste-card">
                <div class="taste-card-label">
                    Favorite Decade
                </div>
                <div class="taste-card-value">
                    {favorite_decade}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="taste-card">
                <div class="taste-card-label">
                    Average Runtime
                </div>
                <div class="taste-card-value">
                    {average_runtime_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # =====================================================
    # TOP GENRES + TOP DIRECTORS
    # =====================================================

    left, right = st.columns(2)

    # -----------------------------------------------------
    # TOP GENRES
    # -----------------------------------------------------

    with left:

        st.subheader("Top Genres")

        if "genre_primary" in data.columns:

            genres = (
                data["genre_primary"]
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(10)
                .rename_axis("Genre")
                .reset_index(name="Movies")
            )

            if not genres.empty:

                genre_chart = px.bar(
                    genres,
                    x="Movies",
                    y="Genre",
                    orientation="h",
                    text="Movies",
                    color_discrete_sequence=[GREEN],
                )

                genre_chart.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )

                genre_chart.update_layout(
                    yaxis={
                        "categoryorder": "total ascending"
                    },
                    xaxis_title="Movies",
                    yaxis_title=None,
                )

                genre_chart = style_chart(
                    genre_chart
                )

                st.plotly_chart(
                    genre_chart,
                    width="stretch",
                )

    # -----------------------------------------------------
    # TOP DIRECTORS
    # -----------------------------------------------------

    with right:

        st.subheader("Top Directors")

        if "director" in data.columns:

            directors = (
                data["director"]
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(10)
                .rename_axis("Director")
                .reset_index(name="Movies")
            )

            if not directors.empty:

                director_chart = px.bar(
                    directors,
                    x="Movies",
                    y="Director",
                    orientation="h",
                    text="Movies",
                    color_discrete_sequence=[ORANGE],
                )

                director_chart.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )

                director_chart.update_layout(
                    yaxis={
                        "categoryorder": "total ascending"
                    },
                    xaxis_title="Movies",
                    yaxis_title=None,
                )

                director_chart = style_chart(
                    director_chart
                )

                st.plotly_chart(
                    director_chart,
                    width="stretch",
                )

        # =====================================================
    # GENRE COMBINATIONS
    # =====================================================

    st.subheader("Genre Combinations")

    st.caption(
        "The genre combinations that appear most often in your watched movies."
    )

    genre_combinations = []

    genre_columns = [
        column
        for column in [
            "genre_primary",
            "genre_secondary",
            "genre_tertiary",
        ]
        if column in data.columns
    ]

    for _, row in data.iterrows():

        movie_genres = []

        for column in genre_columns:

            value = row.get(column)

            if pd.isna(value):
                continue

            value = str(value).strip()

            if value:
                movie_genres.append(value)

        # Remove duplicates while preserving genre names
        movie_genres = list(
            dict.fromkeys(movie_genres)
        )

        # Create every pair of genres for the movie
        for i in range(len(movie_genres)):

            for j in range(
                i + 1,
                len(movie_genres),
            ):

                pair = sorted(
                    [
                        movie_genres[i],
                        movie_genres[j],
                    ]
                )

                genre_combinations.append(
                    f"{pair[0]} + {pair[1]}"
                )

    if genre_combinations:

        combinations = (
            pd.Series(
                genre_combinations
            )
            .value_counts()
            .head(12)
            .rename_axis(
                "Genre Combination"
            )
            .reset_index(
                name="Movies"
            )
        )

        combination_chart = px.bar(
            combinations,
            x="Movies",
            y="Genre Combination",
            orientation="h",
            text="Movies",
            color_discrete_sequence=[
                BLUE
            ],
        )

        combination_chart.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        combination_chart.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            },
            xaxis_title="Movies",
            yaxis_title=None,
            height=520,
        )

        combination_chart = style_chart(
            combination_chart
        )

        st.plotly_chart(
            combination_chart,
            width="stretch",
        )

    else:

        st.info(
            "Not enough genre information to analyze genre combinations."
        )

    st.divider()

    

    # =====================================================
    # RUNTIME + DECADE
    # =====================================================

    left, right = st.columns(2)

    # -----------------------------------------------------
    # RUNTIME DISTRIBUTION
    # -----------------------------------------------------

    with left:

        st.subheader(
            "Runtime Distribution"
        )

        runtime_data = (
            data["runtime_numeric"]
            .dropna()
        )

        runtime_data = runtime_data[
            (runtime_data > 0)
            & (runtime_data <= 300)
        ]

        if not runtime_data.empty:

            runtime_df = pd.DataFrame(
                {
                    "Runtime": runtime_data
                }
            )

            runtime_df["Runtime group"] = pd.cut(
                runtime_df["Runtime"],
                bins=[
                    0,
                    60,
                    90,
                    120,
                    150,
                    180,
                    301,
                ],
                labels=[
                    "<60",
                    "60–89",
                    "90–119",
                    "120–149",
                    "150–179",
                    "180+",
                ],
                right=False,
            )

            runtime_distribution = (
                runtime_df["Runtime group"]
                .value_counts(sort=False)
                .rename_axis("Runtime")
                .reset_index(name="Movies")
            )

            runtime_chart = px.bar(
                runtime_distribution,
                x="Runtime",
                y="Movies",
                text="Movies",
                color_discrete_sequence=[BLUE],
            )

            runtime_chart.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            runtime_chart.update_layout(
                xaxis_title="Minutes",
                yaxis_title="Movies",
            )

            runtime_chart = style_chart(
                runtime_chart
            )

            st.plotly_chart(
                runtime_chart,
                width="stretch",
            )

    # -----------------------------------------------------
    # MOVIES BY DECADE
    # -----------------------------------------------------

    with right:

        st.subheader(
            "Movies by Decade"
        )

        decade_data = (
            data["decade"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .rename_axis("Decade")
            .reset_index(name="Movies")
        )

        if not decade_data.empty:

            decade_data["Decade"] = (
                decade_data["Decade"]
                .astype(str)
                + "s"
            )

            decade_chart = px.bar(
                decade_data,
                x="Decade",
                y="Movies",
                text="Movies",
                color_discrete_sequence=[GREEN],
            )

            decade_chart.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            decade_chart.update_layout(
                xaxis_title=None,
                yaxis_title="Movies",
            )

            decade_chart = style_chart(
                decade_chart
            )

            st.plotly_chart(
                decade_chart,
                width="stretch",
            )

    st.divider()

    # =====================================================
    # COUNTRIES + LANGUAGES
    # =====================================================

    left, right = st.columns(2)

    # -----------------------------------------------------
    # COUNTRIES
    # -----------------------------------------------------

    with left:

        st.subheader(
            "Countries Explored"
        )

        if "country_primary" in data.columns:

            countries = (
                data["country_primary"]
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(10)
                .rename_axis("Country")
                .reset_index(name="Movies")
            )

            if not countries.empty:

                country_chart = px.bar(
                    countries,
                    x="Movies",
                    y="Country",
                    orientation="h",
                    text="Movies",
                    color_discrete_sequence=[BLUE],
                )

                country_chart.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )

                country_chart.update_layout(
                    yaxis={
                        "categoryorder": "total ascending"
                    },
                    xaxis_title="Movies",
                    yaxis_title=None,
                )

                country_chart = style_chart(
                    country_chart
                )

                st.plotly_chart(
                    country_chart,
                    width="stretch",
                )

    # -----------------------------------------------------
    # LANGUAGES
    # -----------------------------------------------------

    with right:

        st.subheader(
            "Languages Watched"
        )

        if "original_language" in data.columns:

            languages = (
                data["original_language"]
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(10)
                .rename_axis("Language")
                .reset_index(name="Movies")
            )

            if not languages.empty:

                language_chart = px.bar(
                    languages,
                    x="Movies",
                    y="Language",
                    orientation="h",
                    text="Movies",
                    color_discrete_sequence=[GREEN],
                )

                language_chart.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )

                language_chart.update_layout(
                    yaxis={
                        "categoryorder": "total ascending"
                    },
                    xaxis_title="Movies",
                    yaxis_title=None,
                )

                language_chart = style_chart(
                    language_chart
                )

                st.plotly_chart(
                    language_chart,
                    width="stretch",
                )

    st.divider()

    # =====================================================
    # RATED MOVIES
    # =====================================================

    rated = data[
        data["rating_numeric"].notna()
    ].copy()

    if rated.empty:

        st.info(
            "No Letterboxd ratings were found."
        )

        return

    # =====================================================
    # HIGHEST + LOWEST RATED MOVIES
    # =====================================================

    st.subheader(
        "Your Ratings"
    )

    favorite_movies = (
        rated
        .sort_values(
            [
                "rating_numeric",
                "vote_average",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(10)
        .copy()
    )

    lowest_movies = (
        rated
        .sort_values(
            [
                "rating_numeric",
                "vote_average",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .head(10)
        .copy()
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            "#### Highest Rated Movies"
        )

        favorite_columns = [
            column
            for column in [
                "Name",
                "Year",
                "director",
                "rating_numeric",
            ]
            if column in favorite_movies.columns
        ]

        favorite_display = (
            favorite_movies[
                favorite_columns
            ]
            .rename(
                columns={
                    "Name": "Movie",
                    "director": "Director",
                    "rating_numeric": "Rating",
                }
            )
        )

        st.dataframe(
            favorite_display,
            hide_index=True,
            width="stretch",
        )

    with right:

        st.markdown(
            "#### Lowest Rated Movies"
        )

        lowest_columns = [
            column
            for column in [
                "Name",
                "Year",
                "director",
                "rating_numeric",
            ]
            if column in lowest_movies.columns
        ]

        lowest_display = (
            lowest_movies[
                lowest_columns
            ]
            .rename(
                columns={
                    "Name": "Movie",
                    "director": "Director",
                    "rating_numeric": "Rating",
                }
            )
        )

        st.dataframe(
            lowest_display,
            hide_index=True,
            width="stretch",
        )

    st.divider()

    # =====================================================
    # BEST RATED GENRES + DIRECTORS
    # =====================================================

    left, right = st.columns(2)

    # -----------------------------------------------------
    # BEST RATED GENRES
    # -----------------------------------------------------

    with left:

        st.subheader(
            "Best Rated Genres"
        )

        valid_genres = rated.copy()

        valid_genres["genre_primary"] = (
            valid_genres["genre_primary"]
            .replace("", pd.NA)
        )

        genre_rating = (
            valid_genres
            .dropna(
                subset=[
                    "genre_primary"
                ]
            )
            .groupby(
                "genre_primary"
            )
            .agg(
                Average_Rating=(
                    "rating_numeric",
                    "mean",
                ),
                Movies=(
                    "rating_numeric",
                    "count",
                ),
            )
            .reset_index()
        )

        genre_rating = (
            genre_rating[
                genre_rating["Movies"] >= 5
            ]
            .sort_values(
                [
                    "Average_Rating",
                    "Movies",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(10)
        )

        if not genre_rating.empty:

            genre_rating["Label"] = (
                genre_rating[
                    "Average_Rating"
                ]
                .map(
                    lambda value:
                    f"{value:.2f}"
                )
            )

            genre_rating_chart = px.bar(
                genre_rating,
                x="Average_Rating",
                y="genre_primary",
                orientation="h",
                text="Label",
                color_discrete_sequence=[GREEN],
            )

            genre_rating_chart.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            genre_rating_chart.update_layout(
                yaxis={
                    "categoryorder":
                    "total ascending"
                },
                xaxis_title="Average Rating",
                yaxis_title=None,
            )

            genre_rating_chart.update_xaxes(
                range=[
                    0,
                    5.2,
                ]
            )

            genre_rating_chart = style_chart(
                genre_rating_chart
            )

            st.plotly_chart(
                genre_rating_chart,
                width="stretch",
            )

    # -----------------------------------------------------
    # BEST RATED DIRECTORS
    # -----------------------------------------------------

    with right:

        st.subheader(
            "Best Rated Directors"
        )

        valid_directors = rated.copy()

        valid_directors["director"] = (
            valid_directors["director"]
            .replace("", pd.NA)
        )

        director_rating = (
            valid_directors
            .dropna(
                subset=[
                    "director"
                ]
            )
            .groupby(
                "director"
            )
            .agg(
                Average_Rating=(
                    "rating_numeric",
                    "mean",
                ),
                Movies=(
                    "rating_numeric",
                    "count",
                ),
            )
            .reset_index()
        )

        director_rating = (
            director_rating[
                director_rating["Movies"] >= 3
            ]
            .sort_values(
                [
                    "Average_Rating",
                    "Movies",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(10)
        )

        if not director_rating.empty:

            director_rating["Label"] = (
                director_rating[
                    "Average_Rating"
                ]
                .map(
                    lambda value:
                    f"{value:.2f}"
                )
            )

            director_rating_chart = px.bar(
                director_rating,
                x="Average_Rating",
                y="director",
                orientation="h",
                text="Label",
                color_discrete_sequence=[ORANGE],
            )

            director_rating_chart.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            director_rating_chart.update_layout(
                yaxis={
                    "categoryorder":
                    "total ascending"
                },
                xaxis_title="Average Rating",
                yaxis_title=None,
            )

            director_rating_chart.update_xaxes(
                range=[
                    0,
                    5.2,
                ]
            )

            director_rating_chart = style_chart(
                director_rating_chart
            )

            st.plotly_chart(
                director_rating_chart,
                width="stretch",
            )

# =========================================================
# YOU VS CROWD
# =========================================================

def render_you_vs_crowd(
    enriched,
    ratings,
):

    st.header("You vs Crowd")

    st.caption(
        "See where your ratings align with — or differ from — TMDB users."
    )

    # =====================================================
    # PREPARE DATA
    # =====================================================

    data = enriched.copy()

    if (
        "vote_average" not in data.columns
        or data["vote_average"].isna().all()
    ):

        st.info(
            "TMDB ratings are not available for enough "
            "movies to build this comparison."
        )

        return

    if ratings.empty:
        st.info("No Letterboxd ratings were found.")
        return

    rating_data = ratings.copy()

    rating_data["Rating"] = pd.to_numeric(
        rating_data["Rating"],
        errors="coerce",
    )

    rating_data = (
        rating_data[
            [
                "Letterboxd URI",
                "Rating",
            ]
        ]
        .drop_duplicates(
            subset=["Letterboxd URI"]
        )
        .rename(
            columns={
                "Rating": "Your Rating"
            }
        )
    )

    data = data.merge(
        rating_data,
        on="Letterboxd URI",
        how="left",
    )

    # TMDB = 0–10
    # Letterboxd = 0–5

    data["TMDB Rating"] = (
        pd.to_numeric(
            data.get(
                "vote_average",
                pd.Series(
                    index=data.index,
                    dtype=float,
                ),
            ),
            errors="coerce",
        )
        / 2
    )

    data["Your Rating"] = pd.to_numeric(
        data["Your Rating"],
        errors="coerce",
    )

    comparison = data[
        data["Your Rating"].notna()
        & data["TMDB Rating"].notna()
        & (data["TMDB Rating"] > 0)
    ].copy()

    if comparison.empty:
        st.info(
            "Not enough rating information is available for comparison."
        )
        return

    comparison["Difference"] = (
        comparison["Your Rating"]
        - comparison["TMDB Rating"]
    )

    comparison["Absolute Difference"] = (
        comparison["Difference"].abs()
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    your_average = comparison[
        "Your Rating"
    ].mean()

    crowd_average = comparison[
        "TMDB Rating"
    ].mean()

    average_difference = comparison[
        "Difference"
    ].mean()

    if average_difference > 0.05:

        rating_style = (
            "More generous<br>"
            "<span class='crowd-card-small'>"
            "than the crowd"
            "</span>"
        )

    elif average_difference < -0.05:

        rating_style = (
            "More critical<br>"
            "<span class='crowd-card-small'>"
            "than the crowd"
            "</span>"
        )

    else:

        rating_style = (
            "Very similar<br>"
            "<span class='crowd-card-small'>"
            "to the crowd"
            "</span>"
        )

    # =====================================================
    # KPI CARDS
    # =====================================================

    st.markdown(
        """
        <style>

        .crowd-card {
            background-color: #151A1E;
            border: 1px solid #303A43;
            border-radius: 10px;
            padding: 18px;
            min-height: 115px;
        }

        .crowd-card-label {
            color: #A8B3BD;
            font-size: 14px;
            margin-bottom: 8px;
        }

        .crowd-card-value {
            color: #FFFFFF;
            font-size: 25px;
            font-weight: 600;
            line-height: 1.15;
        }

        .crowd-card-small {
            color: #A8B3BD;
            font-size: 16px;
            font-weight: 400;
        }

        .chart-legend {
            color: #A8B3BD;
            font-size: 14px;
            margin-top: -5px;
            margin-bottom: 15px;
        }

        .legend-green {
            color: #00E054;
            font-weight: 700;
        }

        .legend-orange {
            color: #FF8000;
            font-weight: 700;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="crowd-card">
                <div class="crowd-card-label">
                    Your Average
                </div>
                <div class="crowd-card-value">
                    {your_average:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="crowd-card">
                <div class="crowd-card-label">
                    Crowd Average
                </div>
                <div class="crowd-card-value">
                    {crowd_average:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="crowd-card">
                <div class="crowd-card-label">
                    Average Difference
                </div>
                <div class="crowd-card-value">
                    {average_difference:+.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        st.markdown(
            f"""
            <div class="crowd-card">
                <div class="crowd-card-label">
                    Rating Style
                </div>
                <div class="crowd-card-value">
                    {rating_style}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # =====================================================
    # YOUR TASTE VS CROWD
    # =====================================================

    st.subheader(
        "Your Taste vs the Crowd"
    )

    st.caption(
        "Average rating by genre. "
        "Genres require at least 5 of your ratings."
    )

    # Legend OUTSIDE the Plotly chart

    st.markdown(
        """
        <div class="chart-legend">
            <span class="legend-green">■</span>
            Your Rating
            &nbsp;&nbsp;&nbsp;&nbsp;
            <span class="legend-orange">■</span>
            TMDB Rating
        </div>
        """,
        unsafe_allow_html=True,
    )

    genre_data = comparison.copy()

    if "genre_primary" in genre_data.columns:

        genre_data["genre_primary"] = (
            genre_data["genre_primary"]
            .replace("", pd.NA)
        )

        genre_comparison = (
            genre_data
            .dropna(
                subset=["genre_primary"]
            )
            .groupby(
                "genre_primary"
            )
            .agg(
                Your_Rating=(
                    "Your Rating",
                    "mean",
                ),
                Crowd_Rating=(
                    "TMDB Rating",
                    "mean",
                ),
                Movies=(
                    "Your Rating",
                    "count",
                ),
            )
            .reset_index()
        )

        genre_comparison = (
            genre_comparison[
                genre_comparison["Movies"] >= 5
            ]
            .sort_values(
                "Movies",
                ascending=False,
            )
            .head(10)
        )

        if not genre_comparison.empty:

            # Sort genres by your rating

            genre_order = (
                genre_comparison
                .sort_values(
                    "Your_Rating",
                    ascending=False,
                )["genre_primary"]
                .tolist()
            )

            genre_long = (
                genre_comparison
                .melt(
                    id_vars=[
                        "genre_primary",
                        "Movies",
                    ],
                    value_vars=[
                        "Your_Rating",
                        "Crowd_Rating",
                    ],
                    var_name="Source",
                    value_name="Rating",
                )
            )

            genre_long["Source"] = (
                genre_long["Source"]
                .replace(
                    {
                        "Your_Rating":
                        "Your Rating",

                        "Crowd_Rating":
                        "TMDB Rating",
                    }
                )
            )

            genre_long["Label"] = (
                genre_long["Rating"]
                .map(
                    lambda value:
                    f"{value:.2f}"
                )
            )

            # VERTICAL BARS

            genre_chart = px.bar(
                genre_long,
                x="genre_primary",
                y="Rating",
                color="Source",
                barmode="group",
                text="Label",
                category_orders={
                    "genre_primary":
                    genre_order
                },
                color_discrete_map={
                    "Your Rating": GREEN,
                    "TMDB Rating": ORANGE,
                },
            )

            genre_chart.update_traces(
                textposition="outside",
                cliponaxis=False,
                textfont={
                    "color": TEXT,
                    "size": 11,
                },
            )

            genre_chart.update_layout(
                xaxis_title=None,
                yaxis_title="Average Rating",
                height=540,

                # Plotly legend removed because
                # we created our own outside.
                showlegend=False,

                bargap=0.22,
                bargroupgap=0.08,

                margin={
                    "l": 30,
                    "r": 25,
                    "t": 35,
                    "b": 90,
                },
            )

            genre_chart.update_xaxes(
                tickangle=-35,
                automargin=True,
            )

            genre_chart.update_yaxes(
                range=[
                    0,
                    5.3,
                ],
                dtick=1,
            )

            genre_chart = style_chart(
                genre_chart,
                show_legend=False,
            )

            st.plotly_chart(
                genre_chart,
                width="stretch",
            )

    st.divider()

    # =====================================================
    # BIGGEST DISAGREEMENTS
    # =====================================================

    st.subheader(
        "Where You Disagree the Most"
    )

    st.caption(
        "Movies with the largest difference between "
        "your rating and the TMDB audience rating."
    )

    disagreements = (
        comparison
        .sort_values(
            "Absolute Difference",
            ascending=False,
        )
        .head(12)
        .copy()
    )

    disagreement_display = (
        disagreements[
            [
                "Name",
                "Year",
                "Your Rating",
                "TMDB Rating",
                "Difference",
            ]
        ]
        .rename(
            columns={
                "Name": "Movie",
            }
        )
        .copy()
    )

    disagreement_display[
        "Your Rating"
    ] = (
        disagreement_display[
            "Your Rating"
        ]
        .round(1)
    )

    disagreement_display[
        "TMDB Rating"
    ] = (
        disagreement_display[
            "TMDB Rating"
        ]
        .round(2)
    )

    disagreement_display[
        "Difference"
    ] = (
        disagreement_display[
            "Difference"
        ]
        .map(
            lambda value:
            f"{value:+.2f}"
        )
    )

    st.dataframe(
        disagreement_display,
        hide_index=True,
        width="stretch",
    )

    st.divider()

    # =====================================================
    # YOU LIKED MORE / LESS
    # =====================================================

    left, right = st.columns(2)

    # -----------------------------------------------------
    # YOU LIKED MORE
    # -----------------------------------------------------

    with left:

        st.subheader(
            "You Liked More"
        )

        st.caption(
            "Movies you rated higher than the TMDB audience."
        )

        # Legend / explanation outside chart

        st.markdown(
            """
            <div class="chart-legend">
                <span class="legend-green">■</span>
                Difference from TMDB rating
            </div>
            """,
            unsafe_allow_html=True,
        )

        liked_more = (
            comparison[
                comparison["Difference"] > 0
            ]
            .sort_values(
                "Difference",
                ascending=False,
            )
            .head(8)
            .copy()
        )

        if not liked_more.empty:

            liked_more[
                "Difference Label"
            ] = (
                liked_more[
                    "Difference"
                ]
                .map(
                    lambda value:
                    f"+{value:.2f}"
                )
            )

            liked_chart = px.bar(
                liked_more,
                x="Difference",
                y="Name",
                orientation="h",
                text="Difference Label",
                color_discrete_sequence=[
                    GREEN
                ],
            )

            liked_chart.update_traces(
                textposition="inside",
                insidetextanchor="end",
                textfont={
                    "color": TEXT,
                    "size": 12,
                },
            )

            liked_chart.update_layout(
                showlegend=False,
                xaxis_title=
                "Rating Difference",
                yaxis_title=None,
                height=470,

                margin={
                    # Extra room for movie names
                    "l": 170,
                    "r": 20,
                    "t": 10,
                    "b": 40,
                },
            )

            liked_chart.update_yaxes(
                categoryorder=
                "total ascending",
                automargin=True,
            )

            liked_chart = style_chart(
                liked_chart,
                show_legend=False,
            )

            st.plotly_chart(
                liked_chart,
                width="stretch",
            )

    # -----------------------------------------------------
    # YOU LIKED LESS
    # -----------------------------------------------------

    with right:

        st.subheader(
            "You Liked Less"
        )

        st.caption(
            "Movies you rated lower than the TMDB audience."
        )

        # Legend / explanation outside chart

        st.markdown(
            """
            <div class="chart-legend">
                <span class="legend-orange">■</span>
                Difference from TMDB rating
            </div>
            """,
            unsafe_allow_html=True,
        )

        liked_less = (
            comparison[
                comparison["Difference"] < 0
            ]
            .sort_values(
                "Difference",
                ascending=True,
            )
            .head(8)
            .copy()
        )

        if not liked_less.empty:

            liked_less[
                "Difference Magnitude"
            ] = (
                liked_less[
                    "Difference"
                ]
                .abs()
            )

            liked_less[
                "Difference Label"
            ] = (
                liked_less[
                    "Difference"
                ]
                .map(
                    lambda value:
                    f"{value:.2f}"
                )
            )

            disliked_chart = px.bar(
                liked_less,
                x="Difference Magnitude",
                y="Name",
                orientation="h",
                text="Difference Label",
                color_discrete_sequence=[
                    ORANGE
                ],
            )

            disliked_chart.update_traces(
                textposition="inside",
                insidetextanchor="end",
                textfont={
                    "color": TEXT,
                    "size": 12,
                },
            )

            disliked_chart.update_layout(
                showlegend=False,
                xaxis_title=
                "Rating Difference",
                yaxis_title=None,
                height=470,

                margin={
                    # Extra room for movie names
                    "l": 170,
                    "r": 20,
                    "t": 10,
                    "b": 40,
                },
            )

            disliked_chart.update_yaxes(
                categoryorder=
                "total ascending",
                automargin=True,
            )

            disliked_chart = style_chart(
                disliked_chart,
                show_legend=False,
            )

            st.plotly_chart(
                disliked_chart,
                width="stretch",
            )

# =========================================================
# RECOMMENDATIONS
# =========================================================

def render_recommendations(
    recommendations,
    profile,
):

    st.header("Recommendations")

    st.caption(
        "Movies selected from your taste profile based on genres, "
        "directors, countries, languages and decades."
    )

    if recommendations is None or recommendations.empty:
        st.info("No recommendations were found.")
        return

    data = recommendations.copy()

    expected_columns = {
        "title": pd.NA,
        "Year": pd.NA,
        "director": pd.NA,
        "genre_primary": pd.NA,
        "country_primary": pd.NA,
        "vote_average": pd.NA,
        "popularity": pd.NA,
        "poster_path": pd.NA,
        "taste_match_score": pd.NA,
    }

    for column, default_value in expected_columns.items():
        if column not in data.columns:
            data[column] = default_value

    for column in [
        "taste_match_score",
        "vote_average",
        "popularity",
        "Year",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["decade"] = (
        (data["Year"] // 10) * 10
    ).astype("Int64")

    best_match = data["taste_match_score"].max()
    valid_genres = data["genre_primary"].replace("", pd.NA).dropna()
    top_genre = (
        valid_genres.value_counts().index[0]
        if not valid_genres.empty
        else "N/A"
    )
    valid_decades = data["decade"].dropna()
    top_decade = (
        f"{int(valid_decades.value_counts().index[0])}s"
        if not valid_decades.empty
        else "N/A"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommendations", len(data))
    col2.metric(
        "Best Taste Score",
        f"{best_match:.0f}/100" if pd.notna(best_match) else "N/A",
    )
    col3.metric("Most Recommended Genre", top_genre)
    col4.metric("Most Recommended Decade", top_decade)

    st.divider()
    st.subheader("Top Recommendations")
    st.caption(
        "Taste Score is a content-based recommendation heuristic, "
        "not a probability."
    )

    top_recommendations = (
        data.sort_values(
            ["taste_match_score", "vote_average"],
            ascending=[False, False],
            na_position="last",
        )
        .head(12)
        .copy()
    )

    for start_index in range(0, len(top_recommendations), 4):
        row = top_recommendations.iloc[start_index:start_index + 4]
        columns = st.columns(4, gap="medium")

        for column, (_, movie) in zip(columns, row.iterrows()):
            with column:
                title = movie.get("title")
                title = (
                    str(title)
                    if pd.notna(title) and str(title).strip()
                    else "Unknown Movie"
                )

                year = movie.get("Year")
                year_text = str(int(year)) if pd.notna(year) else ""

                genre = movie.get("genre_primary")
                genre_text = (
                    str(genre)
                    if pd.notna(genre) and str(genre).strip()
                    else ""
                )

                poster_path = movie.get("poster_path")

                if pd.notna(poster_path) and str(poster_path).strip():
                    st.image(
                        f"https://image.tmdb.org/t/p/w500{poster_path}",
                        width="stretch",
                    )
                else:
                    st.markdown(
                        """
                        <div style="
                            width:100%; aspect-ratio:2/3;
                            background:#1F2A36;
                            border:1px solid #2C3440;
                            border-radius:8px;
                            display:flex; align-items:center;
                            justify-content:center;
                            color:#A8B3BD; font-size:13px;">
                            Poster unavailable
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                info = " · ".join(
                    value for value in [year_text, genre_text] if value
                )
                match = movie.get("taste_match_score")
                tmdb_rating = movie.get("vote_average")

                st.markdown(
                    f'<div class="recommendation-title">{title}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="recommendation-info">{info}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="recommendation-match">'
                    + (
                        f"Taste Score · {float(match):.0f}/100"
                        if pd.notna(match)
                        else "Taste Score unavailable"
                    )
                    + '</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="recommendation-tmdb">'
                    + (
                        f"TMDB {float(tmdb_rating):.1f}/10"
                        if pd.notna(tmdb_rating)
                        else "TMDB rating unavailable"
                    )
                    + '</div>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("Recommendations by Decade")

    decade_data = (
        data["decade"].dropna().astype(int)
        .value_counts().sort_index()
        .rename_axis("Decade")
        .reset_index(name="Movies")
    )

    if not decade_data.empty:
        decade_data["Decade"] = decade_data["Decade"].astype(str) + "s"
        decade_chart = px.bar(
            decade_data,
            x="Decade",
            y="Movies",
            text="Movies",
            color_discrete_sequence=[BLUE],
        )
        decade_chart.update_traces(textposition="outside", cliponaxis=False)
        decade_chart.update_layout(
            showlegend=False,
            xaxis_title=None,
            yaxis_title="Recommendations",
            height=420,
        )
        st.plotly_chart(
            style_chart(decade_chart, show_legend=False),
            width="stretch",
        )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Hidden Gems")
        st.caption("Strong taste scores with lower TMDB popularity.")

        valid_popularity = data["popularity"].dropna()
        if not valid_popularity.empty:
            popularity_median = valid_popularity.median()
            hidden_gems = (
                data[
                    data["popularity"].notna()
                    & (data["popularity"] <= popularity_median)
                    & data["taste_match_score"].notna()
                ]
                .sort_values(
                    ["taste_match_score", "vote_average"],
                    ascending=[False, False],
                    na_position="last",
                )
                .head(8)
                .copy()
            )
        else:
            hidden_gems = pd.DataFrame()

        if not hidden_gems.empty:
            hidden_gems["Score Label"] = hidden_gems["taste_match_score"].map(
                lambda value: f"{value:.0f}/100"
            )
            hidden_chart = px.bar(
                hidden_gems,
                x="taste_match_score",
                y="title",
                orientation="h",
                text="Score Label",
                color_discrete_sequence=[GREEN],
            )
            hidden_chart.update_traces(
                textposition="inside",
                insidetextanchor="end",
                textfont={"color": TEXT, "size": 12},
            )
            hidden_chart.update_xaxes(range=[0, 100])
            hidden_chart.update_yaxes(
                categoryorder="total ascending",
                automargin=True,
            )
            hidden_chart.update_layout(
                showlegend=False,
                xaxis_title="Taste Score",
                yaxis_title=None,
                height=460,
            )
            st.plotly_chart(
                style_chart(hidden_chart, show_legend=False),
                width="stretch",
            )
        else:
            st.info("Not enough popularity data to identify hidden gems.")

    with right:
        st.subheader("Safe Bets")
        st.caption(
            "Strong taste scores that are also highly rated by TMDB users."
        )

        safe_bets = (
            data[
                data["vote_average"].notna()
                & (data["vote_average"] >= 7)
                & data["taste_match_score"].notna()
            ]
            .sort_values(
                ["taste_match_score", "vote_average"],
                ascending=[False, False],
                na_position="last",
            )
            .head(8)
            .copy()
        )

        if not safe_bets.empty:
            safe_bets["Score Label"] = safe_bets["taste_match_score"].map(
                lambda value: f"{value:.0f}/100"
            )
            safe_chart = px.bar(
                safe_bets,
                x="taste_match_score",
                y="title",
                orientation="h",
                text="Score Label",
                color_discrete_sequence=[ORANGE],
            )
            safe_chart.update_traces(
                textposition="inside",
                insidetextanchor="end",
                textfont={"color": TEXT, "size": 12},
            )
            safe_chart.update_xaxes(range=[0, 100])
            safe_chart.update_yaxes(
                categoryorder="total ascending",
                automargin=True,
            )
            safe_chart.update_layout(
                showlegend=False,
                xaxis_title="Taste Score",
                yaxis_title=None,
                height=460,
            )
            st.plotly_chart(
                style_chart(safe_chart, show_legend=False),
                width="stretch",
            )
        else:
            st.info("No highly rated safe bets were found.")

    st.divider()
    st.subheader("Directors Worth Discovering")
    st.caption(
        "Directors appearing repeatedly among your personalized recommendations."
    )

    director_data = data.copy()
    director_data["director"] = director_data["director"].replace("", pd.NA)

    director_summary = (
        director_data.dropna(subset=["director"])
        .groupby("director")
        .agg(
            Recommendations=("title", "count"),
            Average_Match=("taste_match_score", "mean"),
            Average_TMDB=("vote_average", "mean"),
        )
        .reset_index()
        .sort_values(
            ["Recommendations", "Average_Match"],
            ascending=[False, False],
        )
        .head(10)
    )

    if not director_summary.empty:
        director_summary["Label"] = director_summary["Recommendations"].astype(str)
        director_chart = px.bar(
            director_summary,
            x="Recommendations",
            y="director",
            orientation="h",
            text="Label",
            color_discrete_sequence=[BLUE],
        )
        director_chart.update_traces(textposition="outside", cliponaxis=False)
        director_chart.update_yaxes(
            categoryorder="total ascending",
            automargin=True,
        )
        director_chart.update_layout(
            showlegend=False,
            xaxis_title="Recommended Movies",
            yaxis_title=None,
            height=500,
        )
        st.plotly_chart(
            style_chart(director_chart, show_legend=False),
            width="stretch",
        )
    else:
        st.info("Not enough director information is available.")


# =========================================================
# HISTORY
# =========================================================

def render_history(
    enriched,
    ratings,
    likes,
):

    st.header("History")

    st.caption(
        "Search and explore your complete Letterboxd movie history."
    )

    # =====================================================
    # PREPARE DATA
    # =====================================================

    history = attach_user_preferences(
        enriched,
        ratings,
        likes,
    )

    history = history.copy()

    history["rating_numeric"] = pd.to_numeric(
        history.get(
            "rating",
            pd.Series(
                index=history.index,
                dtype=float,
            ),
        ),
        errors="coerce",
    )

    history["year_numeric"] = pd.to_numeric(
        history.get(
            "Year",
            pd.Series(
                index=history.index,
                dtype=float,
            ),
        ),
        errors="coerce",
    )

    history["runtime_numeric"] = pd.to_numeric(
        history.get(
            "runtime_min",
            pd.Series(
                index=history.index,
                dtype=float,
            ),
        ),
        errors="coerce",
    )

    # =====================================================
    # SEARCH
    # =====================================================

    search = st.text_input(
        "Search",
        placeholder="Search by movie or director...",
    )

    # =====================================================
    # FILTER OPTIONS
    # =====================================================

    year_options = []

    if history["year_numeric"].notna().any():

        year_options = (
            history["year_numeric"]
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )

    genre_options = []

    if "genre_primary" in history.columns:

        genre_options = (
            history["genre_primary"]
            .replace("", pd.NA)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )

    country_options = []

    if "country_primary" in history.columns:

        country_options = (
            history["country_primary"]
            .replace("", pd.NA)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )

    # =====================================================
    # FILTERS
    # =====================================================

    filter1, filter2, filter3, filter4 = st.columns(4)

    with filter1:

        selected_genres = st.multiselect(
            "Genre",
            options=genre_options,
        )

    with filter2:

        selected_countries = st.multiselect(
            "Country",
            options=country_options,
        )

    with filter3:

        if year_options:

            min_year = min(year_options)
            max_year = max(year_options)

            selected_years = st.slider(
                "Release year",
                min_value=min_year,
                max_value=max_year,
                value=(
                    min_year,
                    max_year,
                ),
            )

        else:

            selected_years = None

    with filter4:

        rating_filter = st.selectbox(
            "Your rating",
            options=[
                "All",
                "5 stars",
                "4+ stars",
                "3+ stars",
                "Below 3 stars",
                "Not rated",
            ],
        )

    # =====================================================
    # APPLY SEARCH
    # =====================================================

    filtered = history.copy()

    if search:

        search_lower = (
            search
            .strip()
            .lower()
        )

        movie_match = (
            filtered.get(
                "Name",
                pd.Series(
                    "",
                    index=filtered.index,
                ),
            )
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                search_lower,
                regex=False,
            )
        )

        director_match = (
            filtered.get(
                "director",
                pd.Series(
                    "",
                    index=filtered.index,
                ),
            )
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                search_lower,
                regex=False,
            )
        )

        filtered = filtered[
            movie_match
            | director_match
        ]

    # =====================================================
    # APPLY GENRE
    # =====================================================

    if selected_genres:

        filtered = filtered[
            filtered["genre_primary"]
            .isin(
                selected_genres
            )
        ]

    # =====================================================
    # APPLY COUNTRY
    # =====================================================

    if selected_countries:

        filtered = filtered[
            filtered["country_primary"]
            .isin(
                selected_countries
            )
        ]

    # =====================================================
    # APPLY YEAR
    # =====================================================

    if selected_years is not None:

        start_year, end_year = (
            selected_years
        )

        filtered = filtered[
            filtered["year_numeric"]
            .between(
                start_year,
                end_year,
                inclusive="both",
            )
        ]

    # =====================================================
    # APPLY RATING
    # =====================================================

    if rating_filter == "5 stars":

        filtered = filtered[
            filtered["rating_numeric"]
            == 5
        ]

    elif rating_filter == "4+ stars":

        filtered = filtered[
            filtered["rating_numeric"]
            >= 4
        ]

    elif rating_filter == "3+ stars":

        filtered = filtered[
            filtered["rating_numeric"]
            >= 3
        ]

    elif rating_filter == "Below 3 stars":

        filtered = filtered[
            filtered["rating_numeric"]
            < 3
        ]

    elif rating_filter == "Not rated":

        filtered = filtered[
            filtered["rating_numeric"]
            .isna()
        ]

    # =====================================================
    # FILTERED KPIs
    # =====================================================

    st.divider()

    filtered_ratings = (
        filtered["rating_numeric"]
        .dropna()
    )

    filtered_runtime = (
        filtered["runtime_numeric"]
        .dropna()
    )

    average_rating = (
        filtered_ratings.mean()
        if not filtered_ratings.empty
        else None
    )

    total_hours = (
        filtered_runtime.sum()
        / 60
    )

    liked_count = 0

    if "liked" in filtered.columns:

        liked_count = int(
            filtered["liked"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(
        "Movies",
        f"{len(filtered):,}",
    )

    kpi2.metric(
        "Average Rating",
        (
            f"{average_rating:.2f}"
            if average_rating is not None
            else "N/A"
        ),
    )

    kpi3.metric(
        "Total Runtime",
        f"{total_hours:,.0f} h",
    )

    kpi4.metric(
        "Liked",
        f"{liked_count:,}",
    )

    st.divider()

    # =====================================================
    # RESULTS
    # =====================================================

    st.subheader(
        "Movies"
    )

    st.caption(
        f"{len(filtered):,} movies match the current filters."
    )

    # Sort newest release year first by default.

    filtered = filtered.sort_values(
        [
            "year_numeric",
            "Name",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    )

    display_columns = [
        column
        for column in [
            "Name",
            "Year",
            "rating_numeric",
            "liked",
            "director",
            "genre_primary",
            "country_primary",
            "original_language",
            "runtime_numeric",
        ]
        if column in filtered.columns
    ]

    display = (
        filtered[
            display_columns
        ]
        .copy()
        .rename(
            columns={
                "Name": "Movie",
                "rating_numeric": "Rating",
                "liked": "Liked",
                "director": "Director",
                "genre_primary": "Genre",
                "country_primary": "Country",
                "original_language": "Language",
                "runtime_numeric": "Runtime",
            }
        )
    )

    if "Year" in display.columns:

        display["Year"] = (
            pd.to_numeric(
                display["Year"],
                errors="coerce",
            )
            .astype("Int64")
        )

    if "Rating" in display.columns:

        display["Rating"] = (
            pd.to_numeric(
                display["Rating"],
                errors="coerce",
            )
            .round(1)
        )

    if "Runtime" in display.columns:

        display["Runtime"] = (
            pd.to_numeric(
                display["Runtime"],
                errors="coerce",
            )
            .round()
            .astype("Int64")
        )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=650,
        column_config={
            "Movie": st.column_config.TextColumn(
                "Movie",
                width="large",
            ),

            "Year": st.column_config.NumberColumn(
                "Year",
                format="%d",
            ),

            "Rating": st.column_config.NumberColumn(
                "Rating",
                format="%.1f",
            ),

            "Liked": st.column_config.CheckboxColumn(
                "Liked",
            ),

            "Runtime": st.column_config.NumberColumn(
                "Runtime",
                help="Runtime in minutes",
                format="%d min",
            ),
        },
    )

# =========================================================
# MAIN APP
# =========================================================

st.title(
    "Letterboxd Analytics"
)

st.caption(
    "Discover your movie habits, taste profile "
    "and personalized recommendations."
)


uploaded_file = st.file_uploader(
    "Upload your Letterboxd export (.zip)",
    type=[
        "zip"
    ],
)


# =========================================================
# WAITING FOR FILE
# =========================================================

if uploaded_file is None:

    st.markdown(
        """
        Upload your Letterboxd data export to explore:

        - your viewing habits
        - your movie taste
        - how your ratings compare with the crowd
        - personalized movie recommendations
        """
    )


# =========================================================
# PROCESS FILE
# =========================================================

else:

    try:

        upload_bytes = uploaded_file.getvalue()
        upload_fingerprint = hashlib.sha256(
            upload_bytes
        ).hexdigest()
        uploaded_file.seek(0)

        temp_dir = (
            extract_letterboxd_zip(
                uploaded_file
            )
        )

        export_dir = (
            find_export_directory(
                temp_dir
            )
        )

        data = (
            load_letterboxd_data(
                export_dir
            )
        )

        watched = data[
            "watched"
        ]

        ratings = prepare_ratings(
            data[
                "ratings"
            ]
        )

        diary = prepare_diary(
            data[
                "diary"
            ]
        )

        reviews = data[
            "reviews"
        ]

        likes = data[
            "likes"
        ]

        # -------------------------------------------------
        # AUTOMATIC ANALYSIS
        # -------------------------------------------------

        enriched = (
            get_enriched_movies(
                watched,
                upload_fingerprint,
            )
        )

        (
            profile,
            recommendations,
        ) = get_recommendations(
            enriched,
            ratings,
            likes,
        )

        st.success(
            "Your Letterboxd analysis is ready."
        )

        # -------------------------------------------------
        # NAVIGATION
        # -------------------------------------------------

        (
            overview_tab,
            taste_tab,
            crowd_tab,
            recommendations_tab,
            history_tab,
        ) = st.tabs(
            [
                "Overview",
                "Your Taste",
                "You vs Crowd",
                "Recommendations",
                "History",
            ]
        )

        # -------------------------------------------------
        # OVERVIEW
        # -------------------------------------------------

        with overview_tab:

            render_overview(
                watched,
                ratings,
                diary,
                reviews,
                likes,
                enriched,
            )

        # -------------------------------------------------
        # YOUR TASTE
        # -------------------------------------------------

        with taste_tab:

            render_your_taste(
                enriched,
                ratings,
                likes,
                profile,
            )

        # -------------------------------------------------
        # YOU VS CROWD
        # -------------------------------------------------

        with crowd_tab:

            render_you_vs_crowd(
                enriched,
                ratings,
            )

        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------

        with recommendations_tab:

            render_recommendations(
                recommendations,
                profile,
            )

        # -------------------------------------------------
        # HISTORY
        # -------------------------------------------------

        with history_tab:

            render_history(
                enriched,
                ratings,
                likes,
            )


    except Exception as error:

        st.error(
            f"Unable to process this export: {error}"
        )