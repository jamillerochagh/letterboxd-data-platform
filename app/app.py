import hashlib
import html
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    fetch_movie_metadata,
    prefilter_candidates,
    discover_movies_by_directors,
)

from blend_ui import (
    render_blend_invitation,
    render_create_blend,
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Letterboxd Analytics",
    page_icon=":material/movie:",
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

    /* TEXT INPUTS */

    [data-testid="stTextInput"] input {
        background-color:var(--lb-surface-2)!important;
        color:var(--lb-text)!important;
        -webkit-text-fill-color:var(--lb-text)!important;
        caret-color:var(--lb-green)!important;
    }

    [data-testid="stTextInput"] input:focus {
        background-color:var(--lb-surface-2)!important;
        color:var(--lb-text)!important;
        -webkit-text-fill-color:var(--lb-text)!important;
    }

    [data-testid="stTextInput"] input::placeholder {
        color:var(--lb-muted)!important;
        -webkit-text-fill-color:var(--lb-muted)!important;
        opacity:1!important;
    }

    [data-testid="stTextInput"] > div > div {
        background-color:var(--lb-surface-2)!important;
        border-color:var(--lb-border)!important;
    }

    [data-baseweb="input"] {
        background-color:var(--lb-surface-2)!important;
    }

    [data-baseweb="input"] > div {
        background-color:var(--lb-surface-2)!important;
        border-color:var(--lb-border)!important;
    }

    [data-baseweb="select"] > div {
        background:var(--lb-surface-2)!important;
        border-color:var(--lb-border)!important;
        color:var(--lb-text)!important;
    }
    hr { border-color:var(--lb-border)!important; }


    /* GLOBAL PERIOD FILTER */

    .period-heading {
        margin-top:18px;
        margin-bottom:6px;
        color:var(--lb-text);
        font-size:15px;
        font-weight:800;
        letter-spacing:.01em;
    }

    .period-copy {
        color:var(--lb-muted);
        font-size:12px;
        margin-bottom:8px;
    }

    div[data-testid="stSelectbox"]:has(
        select[aria-label="Viewing period"]
    ) {
        background:linear-gradient(
            135deg,
            rgba(0,224,84,.10),
            rgba(64,188,244,.05)
        );
        border:1px solid rgba(0,224,84,.42);
        border-radius:12px;
        padding:12px 14px 14px 14px;
        margin-bottom:18px;
    }

    div[data-testid="stSelectbox"] label {
        color:var(--lb-text)!important;
        font-weight:750!important;
    }

    /* FOCUS / ACTIVE STATES */

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus {
        border-color:var(--lb-green)!important;
        box-shadow:0 0 0 1px var(--lb-green)!important;
        outline:none!important;
    }

    [data-baseweb="tab"]:focus,
    [data-baseweb="tab"]:focus-visible,
    button:focus,
    button:focus-visible {
        outline:none!important;
        box-shadow:0 0 0 1px rgba(0,224,84,.55)!important;
        border-color:var(--lb-green)!important;
    }



    /* STREAMLIT / BASEWEB FOCUS OVERRIDES */

    :root {
        --primary-color:#00E054!important;
    }

    .stTabs [data-baseweb="tab"]:focus,
    .stTabs [data-baseweb="tab"]:focus-visible,
    .stTabs [data-baseweb="tab"][aria-selected="true"]:focus,
    .stTabs [data-baseweb="tab"][aria-selected="true"]:focus-visible {
        outline:none!important;
        box-shadow:inset 0 -3px 0 #00E054!important;
        border-color:transparent!important;
        border-bottom-color:#00E054!important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
        outline:none!important;
        border-color:#00E054!important;
        box-shadow:0 0 0 1px rgba(0,224,84,.70)!important;
    }

    [data-baseweb="tab"] *,
    [data-baseweb="select"] * {
        outline-color:#00E054!important;
    }

    /* COMPACT TABLES */

    .lb-compact-table-wrap {
        width:100%;
        overflow-x:auto;
        border:1px solid var(--lb-border);
        border-radius:10px;
        background:var(--lb-surface);
    }

    .lb-compact-table {
        width:100%;
        border-collapse:collapse;
        min-width:460px;
    }

    .lb-compact-table th {
        background:var(--lb-surface-2);
        color:var(--lb-muted);
        font-size:11px;
        text-transform:uppercase;
        letter-spacing:.06em;
        text-align:left;
        padding:11px 13px;
        border-bottom:1px solid var(--lb-border);
    }

    .lb-compact-table td {
        color:var(--lb-text);
        font-size:13px;
        padding:11px 13px;
        border-bottom:1px solid rgba(48,58,67,.70);
    }

    .lb-compact-table tr:last-child td {
        border-bottom:0;
    }


    .country-ranking-scroll {
        max-height:270px;
        overflow-y:auto;
        border:1px solid var(--lb-border);
        border-radius:10px;
        background:var(--lb-surface);
        scrollbar-width:thin;
        scrollbar-color:#56636F transparent;
    }

    .country-ranking-scroll .lb-compact-table-wrap {
        border:0;
        border-radius:0;
        overflow:visible;
    }

    .country-ranking-scroll .lb-compact-table th {
        position:sticky;
        top:0;
        z-index:2;
    }

    /* RECOMMENDATION CAROUSEL */

    .lb-rec-carousel {
        display:flex;
        gap:16px;
        overflow-x:auto;
        padding:4px 2px 16px 2px;
        scroll-snap-type:x proximity;
        scrollbar-width:thin;
        scrollbar-color:#56636F transparent;
    }

    .lb-rec-card {
        flex:0 0 180px;
        scroll-snap-align:start;
    }

    .lb-rec-poster {
        width:180px;
        aspect-ratio:2/3;
        object-fit:cover;
        border-radius:9px;
        border:1px solid var(--lb-border);
        background:var(--lb-surface-2);
        display:block;
    }

    .lb-rec-poster-missing {
        width:180px;
        aspect-ratio:2/3;
        border-radius:9px;
        border:1px solid var(--lb-border);
        background:var(--lb-surface-2);
        display:flex;
        align-items:center;
        justify-content:center;
        color:var(--lb-muted);
        font-size:12px;
        text-align:center;
    }


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
    progress_callback=None,
) -> pd.DataFrame:

    previous_fingerprint = st.session_state.get(
        "upload_fingerprint"
    )

    if previous_fingerprint != upload_fingerprint:
        st.session_state.pop("enriched_movies", None)
        st.session_state.pop("recommendations", None)
        st.session_state["upload_fingerprint"] = upload_fingerprint

    if "enriched_movies" not in st.session_state:

            enriched = enrich_movies(
                watched,
                progress_callback=progress_callback,
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

RECOMMENDER_VERSION = "v9_multistage"

RECOMMENDER_VERSION = "v17_rating_integrity"

def get_favorite_directors(
    preference_data: pd.DataFrame,
    limit: int = 5,
) -> list[str]:
    if (
        preference_data is None
        or preference_data.empty
        or "director" not in preference_data.columns
    ):
        return []

    data = preference_data.copy()
    data["director"] = data["director"].replace("", pd.NA)
    data["rating_numeric"] = pd.to_numeric(
        data.get(
            "rating",
            pd.Series(index=data.index, dtype=float),
        ),
        errors="coerce",
    )

    data = data.dropna(subset=["director"])
    if data.empty:
        return []

    summary = (
        data.groupby("director")
        .agg(
            watched=("director", "size"),
            avg_rating=("rating_numeric", "mean"),
        )
        .reset_index()
    )
    summary["avg_rating"] = summary["avg_rating"].fillna(0)
    summary["strength"] = (
        summary["watched"].clip(upper=8) * 0.55
        + summary["avg_rating"] * 0.45
    )

    return (
        summary.sort_values(
            ["strength", "watched", "avg_rating"],
            ascending=False,
        )
        .head(limit)["director"]
        .tolist()
    )


def get_recommendations(
    enriched: pd.DataFrame,
    ratings: pd.DataFrame,
    likes: pd.DataFrame,
):
    preference_data = attach_user_preferences(
        enriched,
        ratings,
        likes,
    )
    profile = build_taste_profile(preference_data)

    cache_key = (
        st.session_state.get("upload_fingerprint"),
        RECOMMENDER_VERSION,
    )

    if st.session_state.get("recommendations_cache_key") != cache_key:
        with st.spinner("Building personalized recommendations..."):
            preferred_genres = list(
                profile.get(
                    "genre_combined",
                    profile.get("genre_primary", {}),
                ).keys()
            )[:6]

            genre_candidates = build_candidate_catalog(
                preferred_genres,
                pages_per_genre=2,
            )
            genre_candidates = prefilter_candidates(
                genre_candidates,
                limit=100,
            )

            favorite_directors = get_favorite_directors(
                preference_data,
                limit=5,
            )
            director_candidates = discover_movies_by_directors(
                favorite_directors,
                pages_per_director=1,
            )
            director_candidates = prefilter_candidates(
                director_candidates,
                limit=60,
            )

            candidate_map = {}
            for candidate in genre_candidates + director_candidates:
                movie_id = candidate.get("id")
                if movie_id:
                    candidate_map[movie_id] = candidate

            candidate_df = enrich_candidate_catalog(
                list(candidate_map.values()),
                max_workers=8,
            )

            watched_ids = set(
                pd.to_numeric(
                    enriched.get("tmdb_id", pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna().astype(int)
            )

            recommendations = rank_candidates(
                candidate_df,
                profile,
                watched_ids,
                limit=60,
            )

            st.session_state["recommendations"] = recommendations
            st.session_state["recommendations_cache_key"] = cache_key

    return profile, st.session_state.get(
        "recommendations",
        pd.DataFrame(),
    )


# =========================================================
# OVERVIEW
# =========================================================


def _movie_key_frame(data: pd.DataFrame) -> pd.Series:
    if data is None or data.empty:
        return pd.Series(dtype="string")

    if "Name" in data.columns:
        title_source = data["Name"]
    elif "title" in data.columns:
        title_source = data["title"]
    else:
        title_source = pd.Series(
            "",
            index=data.index,
            dtype="string",
        )

    if "Year" in data.columns:
        year_source = data["Year"]
    elif "release_year" in data.columns:
        year_source = data["release_year"]
    else:
        year_source = pd.Series(
            "",
            index=data.index,
            dtype="string",
        )

    names = (
        title_source
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    years = (
        year_source
        .fillna("")
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )

    return names + "||" + years


def filter_letterboxd_period(
    selected_year,
    watched,
    ratings,
    diary,
    reviews,
    likes,
    enriched,
):
    """
    Filter retrospective analytics to a diary year.
    Recommendations and Movie Blend intentionally remain full-history.
    """
    if selected_year == "All Time":
        return watched, ratings, diary, reviews, likes, enriched

    year = int(selected_year)

    if (
        diary.empty
        or "Watched Date" not in diary.columns
    ):
        return (
            watched.iloc[0:0].copy(),
            ratings.iloc[0:0].copy(),
            diary.iloc[0:0].copy(),
            reviews.iloc[0:0].copy(),
            likes.iloc[0:0].copy(),
            enriched.iloc[0:0].copy(),
        )

    period_diary = diary[
        diary["Watched Date"].dt.year.eq(year)
    ].copy()

    period_keys = set(
        _movie_key_frame(period_diary).tolist()
    )

    def filter_frame(frame):
        if frame is None or frame.empty:
            return frame.copy()

        return frame[
            _movie_key_frame(frame).isin(period_keys)
        ].copy()

    return (
        filter_frame(watched),
        filter_frame(ratings),
        period_diary,
        filter_frame(reviews),
        filter_frame(likes),
        filter_frame(enriched),
    )



def _rating_stars(rating) -> str:
    value = pd.to_numeric(
        pd.Series([rating]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return ""

    value = max(
        0.0,
        min(float(value), 5.0),
    )

    full = int(value)
    half = (
        value - full
    ) >= 0.5

    return (
        "★" * full
        + ("½" if half else "")
    )


def render_compact_table(
    data: pd.DataFrame,
):
    if data is None or data.empty:
        return

    headers = "".join(
        f"<th>{html.escape(str(column))}</th>"
        for column in data.columns
    )

    rows = []

    for _, row in data.iterrows():
        cells = []

        for value in row.tolist():
            if pd.isna(value):
                display_value = ""
            else:
                display_value = str(value)

            cells.append(
                "<td>"
                + html.escape(display_value)
                + "</td>"
            )

        rows.append(
            "<tr>"
            + "".join(cells)
            + "</tr>"
        )

    st.markdown(
        """
        <div class="lb-compact-table-wrap">
            <table class="lb-compact-table">
                <thead><tr>
        """
        + headers
        + """
                </tr></thead>
                <tbody>
        """
        + "".join(rows)
        + """
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )




@st.cache_data(
    show_spinner=False,
    ttl=60 * 60 * 24 * 30,
)
def get_timeline_movie_metadata(
    title: str,
    year: str = "",
):
    try:
        return fetch_movie_metadata(
            title,
            year,
        )
    except Exception:
        return {}



def attach_cached_posters(
    movies: pd.DataFrame,
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Attach already-cached posters without making network requests."""
    if movies is None or movies.empty or enriched is None or enriched.empty:
        return movies

    result = movies.copy()
    source = enriched.copy()

    if "poster_path" not in result.columns:
        result["poster_path"] = pd.NA
    if "poster_path" not in source.columns:
        return result

    valid_source = source[
        source["poster_path"].notna()
        & source["poster_path"].astype(str).str.strip().ne("")
    ].copy()

    if valid_source.empty:
        return result

    missing = (
        result["poster_path"].isna()
        | result["poster_path"].astype(str).str.strip().eq("")
    )

    # 1. TMDB ID is the strongest join key.
    if "tmdb_id" in result.columns and "tmdb_id" in valid_source.columns:
        source_ids = pd.to_numeric(valid_source["tmdb_id"], errors="coerce")
        result_ids = pd.to_numeric(result["tmdb_id"], errors="coerce")
        id_map = (
            valid_source.assign(_id=source_ids)
            .dropna(subset=["_id"])
            .drop_duplicates("_id")
            .set_index("_id")["poster_path"]
            .to_dict()
        )
        result.loc[missing, "poster_path"] = result_ids[missing].map(id_map)
        missing = (
            result["poster_path"].isna()
            | result["poster_path"].astype(str).str.strip().eq("")
        )

    # 2. Letterboxd URI is stable across export files when present.
    uri_candidates = ["Letterboxd URI", "letterboxd_uri", "URI"]
    result_uri = next((c for c in uri_candidates if c in result.columns), None)
    source_uri = next((c for c in uri_candidates if c in valid_source.columns), None)

    if result_uri and source_uri and missing.any():
        uri_map = (
            valid_source.dropna(subset=[source_uri])
            .drop_duplicates(source_uri)
            .set_index(source_uri)["poster_path"]
            .to_dict()
        )
        result.loc[missing, "poster_path"] = result.loc[missing, result_uri].map(uri_map)
        missing = (
            result["poster_path"].isna()
            | result["poster_path"].astype(str).str.strip().eq("")
        )

    # 3. Final fallback: normalized title + release year.
    def make_keys(frame):
        title_col = "Name" if "Name" in frame.columns else "title"
        year_col = "Year" if "Year" in frame.columns else "release_year"

        titles = (
            frame.get(title_col, pd.Series("", index=frame.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        years = pd.to_numeric(
            frame.get(year_col, pd.Series(pd.NA, index=frame.index)),
            errors="coerce",
        ).astype("Int64").astype(str)

        return titles + "||" + years

    if missing.any():
        valid_source["_poster_key"] = make_keys(valid_source)
        result["_poster_key"] = make_keys(result)

        key_map = (
            valid_source.drop_duplicates("_poster_key")
            .set_index("_poster_key")["poster_path"]
            .to_dict()
        )
        result.loc[missing, "poster_path"] = (
            result.loc[missing, "_poster_key"].map(key_map)
        )
        result = result.drop(columns=["_poster_key"], errors="ignore")

    return result


def render_movie_poster_carousel(
    movies: pd.DataFrame,
    empty_message: str,
):
    if movies is None or movies.empty:
        st.caption(empty_message)
        return

    cards = []

    for _, movie in movies.iterrows():
        title = movie.get(
            "Name",
            movie.get(
                "title",
                "Unknown Movie",
            ),
        )

        title = (
            str(title)
            if pd.notna(title)
            else "Unknown Movie"
        )

        poster_path = movie.get(
            "poster_path"
        )


        if (
            poster_path is not None
            and pd.notna(poster_path)
            and str(poster_path).strip()
        ):
            poster_html = (
                '<img class="lb-rec-poster" src="'
                'https://image.tmdb.org/t/p/w500'
                + html.escape(str(poster_path))
                + '" alt="'
                + html.escape(title)
                + '">'
            )
        else:
            poster_html = (
                '<div class="lb-rec-poster-missing">'
                'Poster unavailable'
                '</div>'
            )

        rating = movie.get(
            "rating_numeric",
            movie.get("Rating"),
        )

        stars = _rating_stars(
            rating
        )

        rating_html = (
            '<div class="recommendation-match">'
            + html.escape(stars)
            + '</div>'
            if stars
            else ""
        )

        cards.append(
            '<div class="lb-rec-card">'
            + poster_html
            + '<div class="recommendation-title">'
            + html.escape(title)
            + '</div>'
            + rating_html
            + '</div>'
        )

    st.markdown(
        '<div class="lb-rec-carousel">'
        + "".join(cards)
        + '</div>',
        unsafe_allow_html=True,
    )



def render_country_ranking(
    countries: pd.DataFrame,
):
    if countries is None or countries.empty:
        return

    rows = []

    for index, row in countries.iterrows():
        rows.append(
            "<tr>"
            f"<td>{int(index) + 1}</td>"
            f"<td>{html.escape(str(row['Country']))}</td>"
            f"<td>{int(row['Movies'])}</td>"
            "</tr>"
        )

    st.markdown(
        """
        <div class="country-ranking-scroll">
            <div class="lb-compact-table-wrap">
                <table class="lb-compact-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Country</th>
                            <th>Movies</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        + "".join(rows)
        + """
                    </tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_first_last_watch(
    diary,
    enriched,
    selected_year,
):
    if (
        diary.empty
        or "Watched Date" not in diary.columns
    ):
        return

    dated = (
        diary
        .dropna(subset=["Watched Date"])
        .sort_values("Watched Date")
        .copy()
    )

    if dated.empty:
        return

    metadata = enriched.copy()

    if not metadata.empty:
        metadata["_movie_key"] = _movie_key_frame(
            metadata
        )
        metadata = (
            metadata
            .drop_duplicates("_movie_key")
            .set_index("_movie_key")
        )

    if selected_year == "All Time":
        section_title = "Your Film Timeline"
        first_label = "First watch on your account"
        last_label = "Most recent watch"
    else:
        section_title = f"{selected_year} — First & Last Watch"
        first_label = "First watch"
        last_label = "Last watch"

    st.subheader(section_title)

    def render_card(column, label, row):
        with column:
            key = _movie_key_frame(
                pd.DataFrame([row])
            ).iloc[0]

            poster_path = None

            if (
                not metadata.empty
                and key in metadata.index
            ):
                movie_metadata = metadata.loc[key]
                poster_path = movie_metadata.get(
                    "poster_path"
                )

                if isinstance(
                    poster_path,
                    pd.Series,
                ):
                    poster_path = poster_path.iloc[0]

            if (
                (
                    poster_path is None
                    or pd.isna(poster_path)
                    or not str(poster_path).strip()
                )
                and not enriched.empty
            ):
                row_title = str(
                    row.get("Name", "")
                ).strip().lower()

                if "Name" in enriched.columns:
                    title_series = (
                        enriched["Name"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )
                elif "title" in enriched.columns:
                    title_series = (
                        enriched["title"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )
                else:
                    title_series = pd.Series(
                        "",
                        index=enriched.index,
                    )

                title_matches = enriched[
                    title_series.eq(row_title)
                ]

                if (
                    not title_matches.empty
                    and "poster_path"
                    in title_matches.columns
                ):
                    candidate_poster = (
                        title_matches["poster_path"]
                        .dropna()
                    )

                    if not candidate_poster.empty:
                        poster_path = (
                            candidate_poster.iloc[0]
                        )

            # Final fallback: fetch only these two timeline movies
            # directly from TMDB when the enriched frame has no poster.
            if (
                poster_path is None
                or pd.isna(poster_path)
                or not str(poster_path).strip()
            ):
                try:
                    movie_year = row.get("Year", "")
                    metadata_fallback = get_timeline_movie_metadata(
                        str(row.get("Name", "")),
                        (
                            str(int(float(movie_year)))
                            if pd.notna(movie_year)
                            else ""
                        ),
                    )
                    poster_path = metadata_fallback.get(
                        "poster_path"
                    )
                except Exception:
                    poster_path = None

            inner_left, inner_right = st.columns(
                [1, 1.65],
                gap="medium",
            )

            with inner_left:
                if (
                    poster_path is not None
                    and pd.notna(poster_path)
                    and str(poster_path).strip()
                ):
                    st.image(
                        "https://image.tmdb.org/t/p/w500"
                        f"{poster_path}",
                        width="stretch",
                    )

            with inner_right:
                st.caption(label)

                title = str(
                    row.get(
                        "Name",
                        "Unknown movie",
                    )
                )

                movie_year = row.get("Year")
                year_label = ""

                if pd.notna(movie_year):
                    try:
                        year_label = (
                            f" ({int(float(movie_year))})"
                        )
                    except (TypeError, ValueError):
                        year_label = (
                            f" ({movie_year})"
                        )

                st.markdown(
                    f"**{title}{year_label}**"
                )

                watched_date = row.get(
                    "Watched Date"
                )

                if pd.notna(watched_date):
                    st.caption(
                        pd.Timestamp(
                            watched_date
                        ).strftime(
                            "%B %d, %Y"
                        )
                    )

                stars = _rating_stars(
                    row.get("Rating")
                )

                if stars:
                    st.markdown(
                        f"""
                        <div style="
                            color:#00E054;
                            font-size:20px;
                            letter-spacing:1px;
                            margin-top:5px;
                        ">
                            {stars}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    first_col, last_col = st.columns(2)

    render_card(
        first_col,
        first_label,
        dated.iloc[0],
    )

    render_card(
        last_col,
        last_label,
        dated.iloc[-1],
    )

    st.divider()


def render_overview(
    watched,
    ratings,
    diary,
    reviews,
    likes,
    enriched,
    profile,
    selected_year="All Time",
):

    st.header(
        "Overview"
    )

    if selected_year != "All Time":
        st.caption(
            f"Your {selected_year} in film"
        )

    render_first_last_watch(
        diary,
        enriched,
        selected_year,
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


    st.subheader(
        "Your Taste"
    )

    render_your_taste(
        enriched,
        ratings,
        likes,
        profile,
        embedded=True,
    )


# =========================================================
# YOUR TASTE
# =========================================================


def build_actor_counts(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    if (
        enriched is None
        or enriched.empty
        or "cast_top" not in enriched.columns
    ):
        return pd.DataFrame(
            columns=["Actor", "Movies"]
        )

    actor_counts = (
        enriched["cast_top"]
        .replace("", pd.NA)
        .dropna()
        .astype(str)
        .str.split("|")
        .explode()
        .str.strip()
    )

    actor_counts = actor_counts[
        actor_counts.ne("")
    ]

    if actor_counts.empty:
        return pd.DataFrame(
            columns=["Actor", "Movies"]
        )

    return (
        actor_counts
        .value_counts()
        .head(20)
        .rename_axis("Actor")
        .reset_index(name="Movies")
    )



def validate_rating_integrity(
    preference_data: pd.DataFrame,
    ratings: pd.DataFrame,
):
    """Fail closed if a displayed rating disagrees with Letterboxd's ratings.csv."""
    if (
        preference_data is None
        or preference_data.empty
        or ratings is None
        or ratings.empty
        or "Rating" not in ratings.columns
    ):
        return preference_data

    data = preference_data.copy()

    # Build source-of-truth maps from the current uploaded export.
    source = ratings.copy()
    source["Rating"] = pd.to_numeric(source["Rating"], errors="coerce")

    expected = pd.Series(pd.NA, index=data.index, dtype="Float64")

    if "Letterboxd URI" in data.columns and "Letterboxd URI" in source.columns:
        data_uri = (
            data["Letterboxd URI"].fillna("").astype(str).str.strip().str.rstrip("/")
        )
        source_uri = (
            source["Letterboxd URI"].fillna("").astype(str).str.strip().str.rstrip("/")
        )
        uri_map = (
            source.assign(_uri=source_uri)
            .loc[lambda frame: frame["_uri"].ne("")]
            .drop_duplicates("_uri", keep="last")
            .set_index("_uri")["Rating"]
        )
        expected = pd.to_numeric(data_uri.map(uri_map), errors="coerce")

    # Replace the attached rating with the export's current value.
    data["rating"] = expected
    return data



def render_your_taste(
    enriched,
    ratings,
    likes,
    profile,
    embedded=False,
):

    if not embedded:
        st.header("Your Taste")

    # =====================================================
    # PREPARE DATA
    # =====================================================

    data = attach_user_preferences(
        enriched,
        ratings,
        likes,
    )
    data = validate_rating_integrity(
        data,
        ratings,
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

        st.subheader("Most Watched Genres")

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

        st.subheader("Most Watched Directors")

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
    # ACTORS
    # =====================================================

    st.subheader(
        "Most Watched Actors"
    )

    actor_counts = build_actor_counts(
        data
    )

    cast_coverage = 0.0

    if (
        not data.empty
        and "cast_top" in data.columns
    ):
        cast_coverage = (
            data["cast_top"]
            .replace("", pd.NA)
            .notna()
            .mean()
        )

    if actor_counts.empty or cast_coverage < 0.80:
        with_cast = (
            data["cast_top"].replace("", pd.NA).notna().sum()
            if "cast_top" in data.columns
            else 0
        )
        st.caption(
            f"Actor data loaded for {with_cast:,} of {len(data):,} movies "
            f"in this viewing period ({cast_coverage:.0%})."
        )
    else:
        actor_chart = px.treemap(
            actor_counts,
            path=["Actor"],
            values="Movies",
            color="Movies",
            color_continuous_scale=[
                [0.0, "#3A2412"],
                [0.55, "#B85C00"],
                [1.0, "#FF8000"],
            ],
        )

        actor_chart.update_traces(
            texttemplate=(
                "<b>%{label}</b><br>"
                "%{value} movies"
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "%{value} movies"
                "<extra></extra>"
            ),
        )

        actor_chart.update_layout(
            height=430,
            margin=dict(
                l=0,
                r=0,
                t=5,
                b=0,
            ),
            paper_bgcolor=BACKGROUND,
            plot_bgcolor=BACKGROUND,
            font=dict(
                color=TEXT,
            ),
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            actor_chart,
            width="stretch",
        )

    st.divider()

    # =====================================================
    # COUNTRIES
    # =====================================================

    st.subheader(
        "Your Cinema Map"
    )

    st.caption(
        "Countries represented in the movies you watched."
    )

    if "country_primary" in data.columns:

        countries = (
            data["country_primary"]
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .rename_axis("Country")
            .reset_index(name="Movies")
        )

        if not countries.empty:

            map_col, list_col = st.columns(
                [2.15, 1],
                gap="large",
            )

            with map_col:

                country_chart = px.choropleth(
                    countries,
                    locations="Country",
                    locationmode="country names",
                    color="Movies",
                    hover_name="Country",
                    hover_data={
                        "Movies": True,
                        "Country": False,
                    },
                    color_continuous_scale=[
                        [0.0, "#12301F"],
                        [0.35, "#087D36"],
                        [0.7, "#00B849"],
                        [1.0, "#00E054"],
                    ],
                )

                country_chart.update_geos(
                    showframe=False,
                    showcoastlines=False,
                    showcountries=True,
                    countrycolor="#303A43",
                    showland=True,
                    landcolor="#151A1E",
                    showocean=True,
                    oceancolor="#0B0D10",
                    bgcolor="#0B0D10",
                )

                country_chart.update_layout(
                    height=300,
                    margin=dict(
                        l=0,
                        r=0,
                        t=0,
                        b=0,
                    ),
                    paper_bgcolor="#0B0D10",
                    plot_bgcolor="#0B0D10",
                    font=dict(
                        color=TEXT,
                    ),
                    coloraxis_showscale=False,
                )

                st.plotly_chart(
                    country_chart,
                    width="stretch",
                )

            with list_col:

                st.markdown(
                    "#### Most Watched Countries"
                )

                country_ranking = (
                    countries
                    .reset_index(drop=True)
                    .copy()
                )

                render_country_ranking(
                    country_ranking
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

    five_star_movies = (
        rated[
            rated["rating_numeric"].eq(5.0)
        ]
        .sort_values(
            "vote_average",
            ascending=False,
            na_position="last",
        )
        .copy()
    )

    very_low_rated_movies = (
        rated[
            rated["rating_numeric"].le(1.0)
        ]
        .sort_values(
            "rating_numeric",
            ascending=True,
            na_position="last",
        )
        .copy()
    )

    st.markdown(
        "#### Five-Star Movies"
    )

    st.caption(
        "Every movie you rated 5 out of 5."
    )

    render_movie_poster_carousel(
        five_star_movies,
        "No 5-star movies in this viewing period.",
    )

    st.markdown(
        "#### Lowest Rated Movies"
    )

    st.caption(
        "Movies you rated 1 out of 5 or lower."
    )

    render_movie_poster_carousel(
        very_low_rated_movies,
        "No movies rated 1 or lower in this viewing period.",
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

    render_compact_table(
        disagreement_display
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


def render_recommendation_carousel(
    movies: pd.DataFrame,
):
    if movies is None or movies.empty:
        return

    cards = []

    for _, movie in movies.iterrows():
        title = movie.get("title")
        title = (
            str(title)
            if pd.notna(title)
            and str(title).strip()
            else "Unknown Movie"
        )

        year = movie.get("Year")
        year_text = (
            str(int(year))
            if pd.notna(year)
            else ""
        )

        genre = movie.get("genre_primary")
        genre_text = (
            str(genre)
            if pd.notna(genre)
            and str(genre).strip()
            else ""
        )

        poster_path = movie.get("poster_path")

        if (
            pd.notna(poster_path)
            and str(poster_path).strip()
        ):
            poster_html = (
                '<img class="lb-rec-poster" src="'
                'https://image.tmdb.org/t/p/w500'
                + html.escape(str(poster_path))
                + '" alt="'
                + html.escape(title)
                + '">'
            )
        else:
            poster_html = (
                '<div class="lb-rec-poster-missing">'
                'Poster unavailable'
                '</div>'
            )

        info = " · ".join(
            value
            for value in [
                year_text,
                genre_text,
            ]
            if value
        )

        match = movie.get("taste_match_score")
        tmdb_rating = movie.get("vote_average")

        score_text = (
            f"Taste Score · {float(match):.0f}/100"
            if pd.notna(match)
            else "Taste Score unavailable"
        )

        tmdb_text = (
            f"TMDB {float(tmdb_rating):.1f}/10"
            if pd.notna(tmdb_rating)
            else "TMDB rating unavailable"
        )

        cards.append(
            '<div class="lb-rec-card">'
            + poster_html
            + '<div class="recommendation-title">'
            + html.escape(title)
            + '</div>'
            + '<div class="recommendation-info">'
            + html.escape(info)
            + '</div>'
            + '<div class="recommendation-match">'
            + html.escape(score_text)
            + '</div>'
            + '<div class="recommendation-tmdb">'
            + html.escape(tmdb_text)
            + '</div>'
            + '</div>'
        )

    st.markdown(
        '<div class="lb-rec-carousel">'
        + "".join(cards)
        + '</div>',
        unsafe_allow_html=True,
    )



def normalized_movie_key(title, year):
    title_key = str(title or "").strip().casefold()
    try:
        year_key = str(int(float(year))) if pd.notna(year) else ""
    except (TypeError, ValueError):
        year_key = str(year or "").strip()
    return title_key, year_key


def remove_watched_recommendations(
    recommendations: pd.DataFrame,
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Hard safety filter for already-watched movies using ID plus title/year."""
    if recommendations is None or recommendations.empty:
        return recommendations
    if enriched is None or enriched.empty:
        return recommendations

    result = recommendations.copy()

    watched_ids = set(
        pd.to_numeric(
            enriched.get("tmdb_id", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna().astype(int)
    )

    watched_keys = set()
    for _, movie in enriched.iterrows():
        watched_keys.add(
            normalized_movie_key(
                movie.get("Name", movie.get("title", "")),
                movie.get("Year", movie.get("release_year", "")),
            )
        )

    def already_watched(row):
        tmdb_id = pd.to_numeric(
            pd.Series([row.get("tmdb_id")]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(tmdb_id) and int(tmdb_id) in watched_ids:
            return True

        return normalized_movie_key(
            row.get("title", row.get("Name", "")),
            row.get("Year", row.get("release_year", "")),
        ) in watched_keys

    return result.loc[
        ~result.apply(already_watched, axis=1)
    ].reset_index(drop=True)


def unique_recommendation_rows(
    rows: dict,
    ranked: pd.DataFrame,
    limit: int = 12,
) -> dict:
    """Deduplicate rows and refill each row from its own eligible candidate pool."""
    used = set()
    cleaned = {}

    def movie_identity(movie):
        tmdb_id = pd.to_numeric(
            pd.Series([movie.get("tmdb_id")]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(tmdb_id):
            return ("tmdb", int(tmdb_id))
        return (
            "movie",
            normalized_movie_key(
                movie.get("title", movie.get("Name", "")),
                movie.get("Year", movie.get("release_year", "")),
            ),
        )

    for row_name, candidates in rows.items():
        if candidates is None or candidates.empty:
            continue

        selected = []
        for index, movie in candidates.iterrows():
            key = movie_identity(movie)
            if key in used:
                continue
            used.add(key)
            selected.append(index)
            if len(selected) >= limit:
                break

        if selected:
            cleaned[row_name] = candidates.loc[selected].copy()

    return cleaned


def _recommendation_rows(
    data: pd.DataFrame,
    enriched: pd.DataFrame,
    ratings: pd.DataFrame,
    likes: pd.DataFrame,
):
    ranked = data.sort_values(
        ["recommendation_score", "taste_match_score", "vote_average"],
        ascending=[False, False, False],
        na_position="last",
    ).copy()

    preferences = attach_user_preferences(enriched, ratings, likes)

    rows = {}
    rows["Top Picks for You"] = ranked.copy()

    # Pull from a dedicated TMDB candidate pool for the user's top five directors.
    favorite_directors = get_favorite_directors(
        preferences,
        limit=5,
    )

    director_pool = ranked[
        ranked["director"].isin(favorite_directors)
    ].copy()
    if not director_pool.empty:
        rows["From Directors You Love"] = director_pool

    valid_popularity = ranked["popularity"].dropna()
    if not valid_popularity.empty:
        popularity_cutoff = valid_popularity.quantile(0.55)
        hidden = ranked[
            ranked["popularity"].le(popularity_cutoff)
            & ranked["vote_average"].ge(6.7)
        ].copy()
        if not hidden.empty:
            rows["Hidden Gems for You"] = hidden

    exploration = ranked[
        ranked["taste_match_score"].between(42, 80, inclusive="both")
        & ranked["vote_average"].ge(6.8)
    ].sort_values(
        ["vote_average", "recommendation_score"],
        ascending=[False, False],
    )
    if not exploration.empty:
        rows["Outside Your Comfort Zone"] = exploration

    safe_bets = ranked[
        ranked["vote_average"].ge(7.0)
        & ranked["taste_match_score"].ge(62)
    ].copy()
    if not safe_bets.empty:
        rows["Safe Bets"] = safe_bets

    return unique_recommendation_rows(
        rows,
        ranked,
        limit=12,
    )


def render_recommendations(
    recommendations,
    profile,
    enriched=None,
    ratings=None,
    likes=None,
):
    st.header("Recommendations")
    st.caption(
        "A multi-stage recommendation system: candidate discovery, quality "
        "filtering, personalized ranking and diversity-aware re-ranking."
    )

    if recommendations is None or recommendations.empty:
        st.info("No recommendations were found.")
        return

    data = remove_watched_recommendations(
        recommendations.copy(),
        enriched,
    )
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
        "recommendation_score": pd.NA,
    }
    for column, default_value in expected_columns.items():
        if column not in data.columns:
            data[column] = default_value

    if data.empty:
        st.info("No new unwatched recommendations were found.")
        return

    for column in [
        "taste_match_score",
        "recommendation_score",
        "vote_average",
        "popularity",
        "Year",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    best_match = data["taste_match_score"].max()
    valid_genres = data["genre_primary"].replace("", pd.NA).dropna()
    top_genre = (
        valid_genres.value_counts().index[0]
        if not valid_genres.empty
        else "N/A"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Recommendations", len(data))
    col2.metric(
        "Best Taste Score",
        f"{best_match:.0f}/100" if pd.notna(best_match) else "N/A",
    )
    col3.metric("Most Recommended Genre", top_genre)

    st.divider()

    if enriched is not None and ratings is not None and likes is not None:
        rows = _recommendation_rows(
            data,
            enriched,
            ratings,
            likes,
        )
    else:
        rows = {"Top Picks for You": data.head(12)}

    row_captions = {
        "Top Picks for You":
            "Your strongest overall matches after quality and diversity re-ranking.",
        "From Directors You Love":
            "Unwatched recommendations from directors that appear repeatedly in your taste history.",
        "Hidden Gems for You":
            "Strong personal matches with lower popularity.",
        "Outside Your Comfort Zone":
            "Highly rated films that sit just outside your strongest taste signals.",
        "Safe Bets":
            "Strong personal matches that are also highly rated by TMDB users.",
    }

    for row_title, row_movies in rows.items():
        if row_movies is None or row_movies.empty:
            continue
        st.subheader(row_title)
        st.caption(row_captions.get(row_title, "Personalized recommendations."))
        render_recommendation_carousel(row_movies)
        st.divider()

    st.subheader("Directors Worth Discovering")
    st.caption(
        "A ranking of directors across your strongest unwatched recommendations."
    )

    director_summary = (
        data.assign(
            director=data["director"].replace("", pd.NA)
        )
        .dropna(subset=["director"])
        .groupby("director")
        .agg(
            Recommendations=("title", "count"),
            Average_Taste_Score=("taste_match_score", "mean"),
        )
        .reset_index()
        .sort_values(
            ["Average_Taste_Score", "Recommendations"],
            ascending=[False, False],
        )
        .head(10)
        .sort_values("Average_Taste_Score")
    )

    if not director_summary.empty:
        director_chart = px.bar(
            director_summary,
            x="Average_Taste_Score",
            y="director",
            orientation="h",
            text="Average_Taste_Score",
            hover_data={"Recommendations": True},
        )
        director_chart.update_traces(
            texttemplate="%{text:.0f}",
            textposition="outside",
            cliponaxis=False,
        )
        director_chart.update_layout(
            xaxis_title="Average Taste Score",
            yaxis_title=None,
            showlegend=False,
            height=430,
        )
        st.plotly_chart(
            style_chart(director_chart),
            width="stretch",
        )
    else:
        st.caption(
            "No director data is available for this recommendation set."
        )


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

blend_id = st.query_params.get(
    "blend"
)

if blend_id:

    render_blend_invitation(
        blend_id
    )

    st.stop()

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

        # -------------------------------------------------
        # ANALYSIS PROGRESS
        # -------------------------------------------------

        analysis_progress = st.progress(
            0,
            text="Reading your Letterboxd export... 0%",
        )

        def set_analysis_progress(
            percentage: int,
            message: str,
        ):
            percentage = max(
                0,
                min(int(percentage), 100),
            )

            analysis_progress.progress(
                percentage,
                text=f"{message} {percentage}%",
            )

        # -------------------------------------------------
        # READ UPLOAD
        # -------------------------------------------------

        set_analysis_progress(
            5,
            "Reading your Letterboxd export...",
        )

        upload_bytes = uploaded_file.getvalue()

        upload_fingerprint = hashlib.sha256(
            upload_bytes
        ).hexdigest()

        uploaded_file.seek(0)

        # -------------------------------------------------
        # EXTRACT EXPORT
        # -------------------------------------------------

        set_analysis_progress(
            10,
            "Extracting your Letterboxd data...",
        )

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

        # -------------------------------------------------
        # LOAD DATA
        # -------------------------------------------------

        set_analysis_progress(
            15,
            "Loading your movie history...",
        )

        data = (
            load_letterboxd_data(
                export_dir
            )
        )

        watched = data[
            "watched"
        ]

        # -------------------------------------------------
        # PREPARE DATA
        # -------------------------------------------------

        set_analysis_progress(
            20,
            "Preparing ratings and diary...",
        )

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
        # MOVIE ENRICHMENT
        # -------------------------------------------------

        set_analysis_progress(
            25,
            "Preparing movie data...",
        )

        def update_movie_enrichment(
            movie_progress: float,
        ):
            """
            Movie enrichment represents 25% -> 75%
            of the complete analysis.
            """

            overall_percentage = (
                25
                + int(movie_progress * 50)
            )

            set_analysis_progress(
                overall_percentage,
                "Preparing movie data...",
            )

        enriched = get_enriched_movies(
            watched,
            upload_fingerprint,
            progress_callback=update_movie_enrichment,
        )

        # -------------------------------------------------
        # TASTE + RECOMMENDATIONS
        # -------------------------------------------------

        set_analysis_progress(
            80,
            "Analyzing your movie taste...",
        )

        preference_data = attach_user_preferences(
            enriched,
            ratings,
            likes,
        )
        profile = build_taste_profile(
            preference_data
        )

        # -------------------------------------------------
        # FINALIZE
        # -------------------------------------------------

        set_analysis_progress(
            95,
            "Building your dashboard...",
        )

        set_analysis_progress(
            100,
            "Your Letterboxd analysis is ready.",
        )

        analysis_progress.empty()

        st.success(
            "Your Letterboxd analysis is ready."
        )

        # -------------------------------------------------
        # GLOBAL PERIOD FILTER
        # -------------------------------------------------

        available_years = []

        if (
            not diary.empty
            and "Watched Date" in diary.columns
        ):
            available_years = sorted(
                diary["Watched Date"]
                .dropna()
                .dt.year
                .astype(int)
                .unique()
                .tolist(),
                reverse=True,
            )

        period_options = (
            ["All Time"]
            + [
                str(year)
                for year in available_years
            ]
        )

        st.markdown(
            '<div class="period-heading">Explore your film history</div>'
            '<div class="period-copy">'
            'Switch between your complete profile and a yearly retrospective.'
            '</div>',
            unsafe_allow_html=True,
        )

        selected_year = st.selectbox(
            "Viewing period",
            period_options,
            index=0,
            key="dashboard_period",
        )

        (
            period_watched,
            period_ratings,
            period_diary,
            period_reviews,
            period_likes,
            period_enriched,
        ) = filter_letterboxd_period(
            selected_year,
            watched,
            ratings,
            diary,
            reviews,
            likes,
            enriched,
        )

        if selected_year == "All Time":
            period_profile = profile
        else:
            period_preferences = attach_user_preferences(
                period_enriched,
                period_ratings,
                period_likes,
            )

            period_profile = build_taste_profile(
                period_preferences
            )

        # -------------------------------------------------
        # NAVIGATION — lazy rendering
        # -------------------------------------------------

        section = st.segmented_control(
            "Dashboard section",
            options=[
                "Overview",
                "You vs Crowd",
                "Recommendations",
                "Movie Blend",
            ],
            default="Overview",
            key="main_navigation_v12",
            label_visibility="collapsed",
        )

        if section == "Overview":
            render_overview(
                period_watched,
                period_ratings,
                period_diary,
                period_reviews,
                period_likes,
                period_enriched,
                period_profile,
                selected_year=selected_year,
            )

        elif section == "You vs Crowd":
            render_you_vs_crowd(
                period_enriched,
                period_ratings,
            )

        elif section == "Recommendations":
            with st.spinner("Building your personalized recommendations..."):
                recommendation_profile, recommendations = get_recommendations(
                    enriched,
                    ratings,
                    likes,
                )

            render_recommendations(
                recommendations,
                recommendation_profile,
                enriched=enriched,
                ratings=ratings,
                likes=likes,
            )

        elif section == "Movie Blend":
            render_create_blend(
                enriched,
                ratings,
                likes,
            )

    except Exception as error:

        st.error(
            f"Unable to process this export: {error}"
        )