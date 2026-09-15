"""
Letterboxd Complete Analysis Pipeline
--------------------------------------
Single script that:
  1. Takes your Letterboxd export folder
  2. Enriches with TMDB data (English)
  3. Builds taste profile
  4. Generates scored history + recommendations

Outputs:
  - letterboxd_enriched.csv           → enriched watch history
  - letterboxd_enriched_scored.csv    → history with taste scores
  - letterboxd_recommendations.csv    → personalized recommendations

How to use:
  1. Export your Letterboxd data (Settings → Import & Export → Export Your Data)
  2. Extract the ZIP file somewhere
  3. Create a .env file with: TMDB_API_KEY=your_key_here
  4. Run:
       pip install requests python-dotenv pandas tqdm
       python letterboxd_analysis.py
  5. When prompted, enter the path to your extracted Letterboxd folder
"""

import csv
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

# Output files
ENRICHED_FILE = "letterboxd_enriched.csv"
CATALOG_FILE = "tmdb_catalog.csv"
OUTPUT_SCORED = "letterboxd_enriched_scored.csv"
OUTPUT_RECS = "letterboxd_recommendations.csv"

# Recommendation settings
PAGES_PER_GENRE = 8        # 8 pages × 20 films = ~160 per genre
MIN_RATING = 3.5           # minimum rating for "liked"
LIKED_BOOST = 1.5          # extra weight for films with liked=Yes
SCORE_THRESHOLD = 20       # minimum score to enter recommendations
MAX_WORKERS = 5            # parallel requests (don't increase too much to avoid rate limit)

WEIGHTS = {
    "genre_primary": 0.30,
    "genre_secondary": 0.15,
    "director": 0.20,
    "country_primary": 0.10,
    "original_language": 0.10,
    "decade": 0.15,
}

# TMDB genres — ids mapped to English names
GENRES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

GENRE_ID_TO_NAME = GENRES

# Renames columns so Tableau doesn't auto-aggregate
TABLEAU_RENAME = {
    "Name": "Title",
    "Year": "Year",
    "rating": "Your Rating",
    "liked": "Liked",
    "rewatch": "Rewatch",
    "tags": "Tags",
    "has_review": "Has Review",
    "watched_date": "Watch Date",
    "director": "Director",
    "genre_primary": "Primary Genre",
    "genre_secondary": "Secondary Genre",
    "genre_tertiary": "Tertiary Genre",
    "country_primary": "Production Country",
    "original_language": "Original Language",
    "runtime_min": "Runtime (min)",
    "vote_average": "TMDB Rating",
    "popularity": "TMDB Popularity",
    "tagline": "Tagline",
    "overview": "Overview",
    "tmdb_id": "TMDB ID",
    "taste_match_score": "Taste Match Score",
    "divergence_score": "Divergence Score",
    "decade": "Decade",
    "review": "Review",
    "tmdb_rating_scaled": "TMDB Rating (0-5)",
}


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def read_csv_file(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_ratings_map(rows: list[dict]) -> dict:
    """Letterboxd URI → rating"""
    return {r["Letterboxd URI"]: r.get("Rating", "") for r in rows}


def build_diary_map(rows: list[dict]) -> dict:
    """Letterboxd URI → {watched_date, rewatch, tags}"""
    result = {}
    for r in rows:
        uri = r["Letterboxd URI"]
        if uri not in result:
            result[uri] = {
                "watched_date": r.get("Watched Date", ""),
                "rewatch": r.get("Rewatch", ""),
                "tags": r.get("Tags", ""),
            }
    return result


def build_reviews_set(rows: list[dict]) -> set:
    """Set of URIs that have reviews"""
    return {r["Letterboxd URI"] for r in rows}


def build_likes_set(rows: list[dict]) -> set:
    """Set of URIs you liked"""
    return {r["Letterboxd URI"] for r in rows}


def search_movie(title: str, year: str) -> Optional[dict]:
    params = {"api_key": TMDB_API_KEY, "query": title, "language": "en-US"}
    if year:
        params["year"] = year
    r = requests.get(f"{BASE_URL}/search/movie", params=params, timeout=10)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results and year:
        params.pop("year")
        r = requests.get(f"{BASE_URL}/search/movie", params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
    return results[0] if results else None


def get_details(movie_id: int) -> dict:
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits",
        "language": "en-US",
    }
    r = requests.get(f"{BASE_URL}/movie/{movie_id}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def empty_tmdb() -> dict:
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
        "tagline": "",
        "overview": "",
    }


def fetch_tmdb(title: str, year: str) -> dict:
    result = search_movie(title, year)
    if not result:
        return empty_tmdb()

    details = get_details(result["id"])
    credits = details.get("credits", {})

    directors = [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"]
    genres = [g["name"] for g in details.get("genres", [])]
    countries = [c["name"] for c in details.get("production_countries", [])]

    return {
        "tmdb_id": details.get("id", ""),
        "director": directors[0] if directors else "",
        "genre_primary": genres[0] if len(genres) > 0 else "",
        "genre_secondary": genres[1] if len(genres) > 1 else "",
        "genre_tertiary": genres[2] if len(genres) > 2 else "",
        "country_primary": countries[0] if countries else "",
        "original_language": details.get("original_language", ""),
        "runtime_min": details.get("runtime", ""),
        "vote_average": details.get("vote_average", ""),
        "popularity": details.get("popularity", ""),
        "tagline": details.get("tagline", ""),
        "overview": details.get("overview", ""),
    }


def enrich_letterboxd_data(raw_dir: str) -> str:
    """Enriches Letterboxd export with TMDB data. Returns path to enriched CSV."""
    print("\n📂 Reading Letterboxd files...")

    watched = read_csv_file(os.path.join(raw_dir, "watched.csv"))
    ratings = build_ratings_map(read_csv_file(os.path.join(raw_dir, "ratings.csv")))
    diary = build_diary_map(read_csv_file(os.path.join(raw_dir, "diary.csv")))
    reviews = build_reviews_set(read_csv_file(os.path.join(raw_dir, "reviews.csv")))
    
    # Handle likes folder structure
    likes_path = os.path.join(raw_dir, "likes", "films.csv")
    if not os.path.exists(likes_path):
        likes_path = os.path.join(raw_dir, "films.csv")
    likes = build_likes_set(read_csv_file(likes_path))

    # Incremental cache from previous enrichment
    existing_map = {}

    if os.path.exists(ENRICHED_FILE):
        try:
            cached = pd.read_csv(ENRICHED_FILE, dtype=str)

            if "Letterboxd URI" in cached.columns:
                existing_map = {
                    row["Letterboxd URI"]: row.to_dict()
                    for _, row in cached.iterrows()
                }

            print(f"📁 Found {len(existing_map)} cached films")
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")

    total = len(watched)
    print(f"🎬 {total} films found. Fetching from TMDB...\n")

    enriched = []
    for i, row in enumerate(watched, 1):
        title = row.get("Name", "").strip()
        year = row.get("Year", "").strip()
        uri = row.get("Letterboxd URI", "")

        # Reuse existing TMDB data
        if uri in existing_map:
            cached_row = existing_map[uri]

            diary_info = diary.get(uri, {})

            internal = {
                "rating": ratings.get(uri, ""),
                "watched_date": diary_info.get("watched_date", ""),
                "rewatch": diary_info.get("rewatch", ""),
                "tags": diary_info.get("tags", ""),
                "has_review": "Yes" if uri in reviews else "No",
                "liked": "Yes" if uri in likes else "No",
            }

            tmdb_fields = {
                "tmdb_id": cached_row.get("tmdb_id", ""),
                "director": cached_row.get("director", ""),
                "genre_primary": cached_row.get("genre_primary", ""),
                "genre_secondary": cached_row.get("genre_secondary", ""),
                "genre_tertiary": cached_row.get("genre_tertiary", ""),
                "country_primary": cached_row.get("country_primary", ""),
                "original_language": cached_row.get("original_language", ""),
                "runtime_min": cached_row.get("runtime_min", ""),
                "vote_average": cached_row.get("vote_average", ""),
                "popularity": cached_row.get("popularity", ""),
                "tagline": cached_row.get("tagline", ""),
                "overview": cached_row.get("overview", ""),
            }

            enriched.append({**row, **internal, **tmdb_fields})
            print(f"[{i}/{total}] ⚡ Cached: {title}")
        continue

        print(f"[{i}/{total}]", end=" ")

        # Internal crossover
        diary_info = diary.get(uri, {})
        internal = {
            "rating": ratings.get(uri, ""),
            "watched_date": diary_info.get("watched_date", ""),
            "rewatch": diary_info.get("rewatch", ""),
            "tags": diary_info.get("tags", ""),
            "has_review": "Yes" if uri in reviews else "No",
            "liked": "Yes" if uri in likes else "No",
        }

        # TMDB
        try:
            tmdb = fetch_tmdb(title, year)
            status = f"{tmdb['director']} | {tmdb['genre_primary']} | {tmdb['country_primary']}"
            print(f"✅ {title} → {status}")
        except Exception as e:
            print(f"❌ Error on '{title}': {e}")
            tmdb = empty_tmdb()

        enriched.append({**row, **internal, **tmdb})
        time.sleep(0.25)

    # Save
    fieldnames = list(enriched[0].keys())

    with open(ENRICHED_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"\n✅ Enrichment complete! Saved to '{ENRICHED_FILE}'")
    print(f"   {len(enriched)} films | {len(fieldnames)} columns")

    return ENRICHED_FILE


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_genre_page(genre_id: int, page: int) -> list[dict]:
    params = {
        "api_key": TMDB_API_KEY,
        "with_genres": genre_id,
        "sort_by": "vote_count.desc",
        "vote_count.gte": 200,
        "page": page,
        "language": "en-US",
        "include_adult": False,
    }
    r = requests.get(f"{BASE_URL}/discover/movie", params=params, timeout=10)
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_movie_details(tmdb_id: int) -> Optional[dict]:
    """Fetches director and country — fields that /discover doesn't return."""
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "append_to_response": "credits",
            "language": "en-US",
        }
        r = requests.get(f"{BASE_URL}/movie/{tmdb_id}", params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
        credits = d.get("credits", {})
        directors = [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"]
        countries = [c["name"] for c in d.get("production_countries", [])]
        return {
            "tmdb_id": tmdb_id,
            "director": directors[0] if directors else "",
            "country_primary": countries[0] if countries else "",
            "tagline": d.get("tagline", ""),
            "overview": d.get("overview", ""),
            "runtime_min": d.get("runtime", ""),
        }
    except Exception:
        return None


def parse_discover_result(movie: dict) -> dict:
    """Extracts fields available directly from /discover (no extra call needed)."""
    genre_ids = movie.get("genre_ids", [])
    genres = [GENRE_ID_TO_NAME.get(gid, "") for gid in genre_ids if gid in GENRE_ID_TO_NAME]
    year_str = (movie.get("release_date", "") or "")[:4]
    return {
        "tmdb_id": movie.get("id", ""),
        "Name": movie.get("title", ""),
        "Year": year_str,
        "original_language": movie.get("original_language", ""),
        "vote_average": movie.get("vote_average", ""),
        "popularity": movie.get("popularity", ""),
        "genre_primary": genres[0] if len(genres) > 0 else "",
        "genre_secondary": genres[1] if len(genres) > 1 else "",
        "genre_tertiary": genres[2] if len(genres) > 2 else "",
        # filled later in detail fetch
        "director": "",
        "country_primary": "",
        "tagline": "",
        "overview": "",
        "runtime_min": "",
    }


def build_catalog(top_genres: set[str]) -> pd.DataFrame:
    """
    Collects films from TMDB.
    2-phase strategy:
      Phase 1 — /discover for all genres (fast, no sleep)
      Phase 2 — /movie/{id} only for films whose genre matches your profile
               (reduces 70–80% of detail calls)
    """
    if os.path.exists(CATALOG_FILE):
        print(f"📁 Catalog already exists ({CATALOG_FILE}). Skipping collection.")
        return pd.read_csv(CATALOG_FILE)

    print("🌐 Phase 1 — collecting film list by genre...")
    seen_ids = set()
    all_movies: list[dict] = []

    genre_items = list(GENRES.items())
    with tqdm(total=len(genre_items) * PAGES_PER_GENRE, unit="pg", ncols=80) as pbar:
        for genre_id, genre_name in genre_items:
            pbar.set_description(f"{genre_name:<20}")
            for page in range(1, PAGES_PER_GENRE + 1):
                try:
                    results = fetch_genre_page(genre_id, page)
                    for movie in results:
                        mid = movie.get("id")
                        if mid and mid not in seen_ids:
                            seen_ids.add(mid)
                            all_movies.append(parse_discover_result(movie))
                except Exception:
                    pass
                pbar.update(1)

    print(f"   {len(all_movies)} unique films found.")

    # Phase 2 — details only for films with relevant genre
    relevant = [m for m in all_movies if m["genre_primary"] in top_genres]
    other = [m for m in all_movies if m["genre_primary"] not in top_genres]
    print(f"\n🔍 Phase 2 — fetching details for {len(relevant)} relevant films "
          f"(skipping {len(other)} outside your profile)...")

    details_map: dict[int, dict] = {}

    def fetch_with_delay(tmdb_id):
        result = fetch_movie_details(tmdb_id)
        time.sleep(0.15)
        return result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_with_delay, m["tmdb_id"]): m["tmdb_id"] for m in relevant}
        with tqdm(total=len(futures), unit="film", ncols=80, desc="Details") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    details_map[result["tmdb_id"]] = result
                pbar.update(1)

    # Merge details
    for movie in all_movies:
        mid = movie["tmdb_id"]
        if mid in details_map:
            movie.update({
                "director": details_map[mid]["director"],
                "country_primary": details_map[mid]["country_primary"],
                "tagline": details_map[mid]["tagline"],
                "overview": details_map[mid]["overview"],
                "runtime_min": details_map[mid]["runtime_min"],
            })

    df = pd.DataFrame(all_movies).drop_duplicates(subset="tmdb_id")
    df.to_csv(CATALOG_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Catalog saved: {CATALOG_FILE} ({len(df)} films)")
    return df


def load_enriched() -> pd.DataFrame:
    # Read everything as string to avoid incorrect automatic conversions
    df = pd.read_csv(ENRICHED_FILE, dtype=str)

    # ── ratings.csv: rating by exact URI (more reliable than enriched) ────────
    ratings_path = "ratings.csv"
    if os.path.exists(ratings_path):
        ratings = pd.read_csv(ratings_path, dtype=str)[["Letterboxd URI", "Rating"]]
        ratings = ratings.rename(columns={"Letterboxd URI": "_uri", "Rating": "_rating"})
        df = df.merge(ratings, left_on="Letterboxd URI", right_on="_uri", how="left")
        # _rating has total priority — overwrites any previous value
        df["rating"] = df["_rating"].combine_first(df["rating"])
        df = df.drop(columns=["_uri", "_rating"])
        filled = pd.to_numeric(df["rating"], errors="coerce").notna().sum()
        print(f"   ⭐ {filled} ratings loaded from ratings.csv")

    # ── diary.csv: watched date, rewatch, tags ────────────────────────────────
    diary_path = "diary.csv"
    if os.path.exists(diary_path):
        diary = pd.read_csv(diary_path, dtype=str)
        diary["key"] = diary["Name"].str.strip().str.lower() + "|" + diary["Year"].str.strip()
        df["key"] = df["Name"].str.strip().str.lower() + "|" + df["Year"].str.strip()
        diary_map = (
            diary
            .sort_values("Watched Date", ascending=False)
            .drop_duplicates(subset="key")
            [["key", "Watched Date", "Rewatch", "Tags"]]
            .rename(columns={"Watched Date": "_wd", "Rewatch": "_rw", "Tags": "_tg"})
        )
        df = df.merge(diary_map, on="key", how="left")
        df["watched_date"] = df["_wd"].fillna(df["watched_date"])
        df["rewatch"] = df["_rw"].fillna(df["rewatch"])
        df["tags"] = df["_tg"].fillna(df["tags"])
        df = df.drop(columns=["key", "_wd", "_rw", "_tg"])
        print(f"   📅 {df['watched_date'].notna().sum()} dates loaded from diary.csv")

    # ── reviews.csv: review text ───────────────────────────────────────────────
    reviews_path = "reviews.csv"
    if os.path.exists(reviews_path):
        reviews = pd.read_csv(reviews_path, dtype=str)
        reviews["key"] = reviews["Name"].str.strip().str.lower() + "|" + reviews["Year"].str.strip()
        df["key"] = df["Name"].str.strip().str.lower() + "|" + df["Year"].str.strip()
        review_map = (
            reviews
            .drop_duplicates(subset="key")
            [["key", "Review"]]
            .rename(columns={"Review": "_review"})
        )
        df = df.merge(review_map, on="key", how="left")
        df["review"] = df.get("_review", "")
        df = df.drop(columns=["key", "_review"])
        print(f"   ✍️  {df['review'].notna().sum()} reviews loaded from reviews.csv")

    # ── types ──────────────────────────────────────────────────────────────────
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce")

    df["decade"] = (df["Year"] // 10 * 10).astype("Int64").astype(str).replace("<NA>", "")
    df["liked"] = df["liked"].fillna("No")
    df["rewatch"] = df["rewatch"].fillna("").str.strip()

    return df


def build_taste_profile(df: pd.DataFrame) -> dict:
    enjoyed = df[df["rating"] >= MIN_RATING].copy()
    enjoyed["weight"] = (enjoyed["rating"] - 0.5) / 4.5
    enjoyed.loc[enjoyed["liked"] == "Yes", "weight"] *= LIKED_BOOST

    profile = {}
    for attr in WEIGHTS:
        if attr not in enjoyed.columns:
            profile[attr] = {}
            continue
        counts = enjoyed.dropna(subset=[attr]).groupby(attr)["weight"].sum()
        profile[attr] = (counts / counts.sum()).to_dict() if counts.sum() > 0 else {}
    return profile


def score_film(row: pd.Series, profile: dict) -> float:
    total = 0.0
    for attr, weight in WEIGHTS.items():
        val = str(row.get(attr, "")).strip()
        if val and val not in ("nan", ""):
            match = profile.get(attr, {}).get(val, 0.0)
            total += match * weight
    return round((total / sum(WEIGHTS.values())) * 100, 2)


def compute_divergence(df: pd.DataFrame) -> pd.Series:
    rated = df["rating"].notna() & df["vote_average"].notna()
    your_norm = (df["rating"] - 0.5) / 4.5
    tmdb_norm = (df["vote_average"] - 1) / 9.0
    return (your_norm - tmdb_norm).abs().mul(100).where(rated).round(2)


def generate_recommendations():
    """Generates taste scores and recommendations."""
    print(f"\n📂 Reading {ENRICHED_FILE}...")
    enriched = load_enriched()
    print(f"   {len(enriched)} films in your history.\n")

    # Build taste profile
    enjoyed_count = (enriched["rating"] >= MIN_RATING).sum()
    print(f"🎯 Building taste profile ({enjoyed_count} films with rating ≥ {MIN_RATING})...")
    profile = build_taste_profile(enriched)
    top_genres = set(list(profile["genre_primary"].keys())[:8])  # top 8 genres

    print("\n   Top genres:")
    for k, v in sorted(profile["genre_primary"].items(), key=lambda x: -x[1])[:5]:
        print(f"     {k}: {v:.1%}")
    print("\n   Top directors:")
    for k, v in sorted(profile["director"].items(), key=lambda x: -x[1])[:5]:
        print(f"     {k}: {v:.1%}")
    print()

    # TMDB catalog
    catalog = build_catalog(top_genres)
    catalog["tmdb_id"] = pd.to_numeric(catalog["tmdb_id"], errors="coerce")
    catalog["Year"] = pd.to_numeric(catalog["Year"], errors="coerce")
    catalog["decade"] = (catalog["Year"] // 10 * 10).astype("Int64").astype(str).replace("<NA>", "")

    # Remove already watched films
    watched_ids = set(enriched["tmdb_id"].dropna().astype(int))
    unseen = catalog[~catalog["tmdb_id"].isin(watched_ids)].copy()
    print(f"\n🎬 {len(catalog)} in catalog | {len(unseen)} not yet watched")

    # Scores on history
    print("\n⚙️  Calculating scores on history...")
    enriched["taste_match_score"] = enriched.apply(lambda r: score_film(r, profile), axis=1)
    enriched["divergence_score"] = compute_divergence(enriched)

    # TMDB normalized to 0–5 scale in 0.5 increments (like Letterboxd)
    enriched["tmdb_rating_scaled"] = (
        (enriched["vote_average"] / 2 * 2).round() / 2
    ).round(1)

    # Ensure numeric columns are saved as float in CSV
    for col in ["rating", "vote_average", "tmdb_rating_scaled", "taste_match_score", "divergence_score"]:
        if col in enriched.columns:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce")

    enriched.rename(columns={k: v for k, v in TABLEAU_RENAME.items() if k in enriched.columns}).to_csv(
        OUTPUT_SCORED, index=False, encoding="utf-8", float_format="%.1f"
    )
    print(f"✅ {OUTPUT_SCORED} saved")

    # Scores on recommendations
    print("⚙️  Calculating scores on recommendations...")
    unseen["taste_match_score"] = unseen.apply(lambda r: score_film(r, profile), axis=1)

    recs = (
        unseen[unseen["taste_match_score"] >= SCORE_THRESHOLD]
        .sort_values("taste_match_score", ascending=False)
        [["Name", "Year", "director", "genre_primary", "genre_secondary",
          "genre_tertiary", "country_primary", "original_language",
          "runtime_min", "vote_average", "popularity",
          "taste_match_score", "tagline", "overview", "tmdb_id", "decade"]]
        .reset_index(drop=True)
        .rename(columns={k: v for k, v in TABLEAU_RENAME.items()})
    )
    recs.to_csv(OUTPUT_RECS, index=False, encoding="utf-8")
    print(f"✅ {OUTPUT_RECS} saved ({len(recs)} recommendations)\n")

    # Preview
    print("🎬 Top 15 recommendations:")
    print(recs[["Title", "Year", "Primary Genre", "Director", "Taste Match Score"]].head(15).to_string(index=False))

    # Divergence stats
    div = enriched.dropna(subset=["divergence_score"])
    above = (enriched["rating"] / 2 > enriched["vote_average"] / 10).sum()
    below = (enriched["rating"] / 2 < enriched["vote_average"] / 10).sum()
    print(f"\n📊 Divergence vs. TMDB ({len(div)} rated films):")
    print(f"   Mean: {div['divergence_score'].mean():.1f} pts")
    print(f"   Above TMDB: {above} | Below: {below}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not TMDB_API_KEY:
        raise ValueError(
            "TMDB_API_KEY not found.\n"
            "Create a .env file in the same folder with:\n"
            "  TMDB_API_KEY=your_key_here"
        )

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║         Letterboxd Complete Analysis Pipeline            ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # Get Letterboxd export folder
    print("📁 Enter the path to your Letterboxd export folder")
    print("   (the folder containing watched.csv, ratings.csv, etc.)")
    print("   Example: /Users/jamille/Downloads/letterboxd-export")
    letterboxd_dir = input("\n→ Path: ").strip().strip('"').strip("'")

    if not os.path.exists(letterboxd_dir):
        print(f"\n❌ Error: folder not found: {letterboxd_dir}")
        return

    watched_path = os.path.join(letterboxd_dir, "watched.csv")
    if not os.path.exists(watched_path):
        print(f"\n❌ Error: watched.csv not found in {letterboxd_dir}")
        print("   Make sure you're pointing to the correct Letterboxd export folder")
        return

    # Step 1: Enrichment
    print("\n" + "═" * 60)
    print("STEP 1: ENRICHING WITH TMDB DATA")
    print("═" * 60)
    enrich_letterboxd_data(letterboxd_dir)

    # Step 2: Recommendations
    print("\n" + "═" * 60)
    print("STEP 2: GENERATING RECOMMENDATIONS")
    print("═" * 60)
    generate_recommendations()

    # Done!
    print("\n" + "═" * 60)
    print("✨ COMPLETE! Import these files to Tableau:")
    print("═" * 60)
    print(f"   📊 {OUTPUT_SCORED}")
    print(f"   🎯 {OUTPUT_RECS}")
    print(f"\n   Your enriched data is also saved in: {ENRICHED_FILE}")


if __name__ == "__main__":
    main()
