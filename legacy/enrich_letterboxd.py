"""
Letterboxd → TMDB Enricher (v3)
--------------------------------
Combina todos os CSVs do export do Letterboxd em um único dataset
enriquecido com dados do TMDB.

Fonte base: watched.csv (todos os filmes assistidos)
Cruzamentos internos:
  - ratings.csv      → sua nota
  - diary.csv        → data assistida, rewatch, tags
  - reviews.csv      → se você escreveu review
  - likes/films.csv  → se você deu like

Colunas adicionadas via TMDB:
  - director          → diretor principal
  - genre_primary     → gênero principal
  - genre_secondary   → segundo gênero
  - genre_tertiary    → terceiro gênero
  - country_primary   → país principal de produção
  - original_language → código do idioma (en, fr, ja...)
  - runtime_min       → duração em minutos
  - vote_average      → nota média do TMDB
  - popularity        → score de popularidade
  - tagline           → frase do filme
  - overview          → sinopse em pt-BR

Como usar:
  1. Extraia o zip do Letterboxd numa pasta chamada letterboxd_raw/
  2. Crie um arquivo .env com: TMDB_API_KEY=sua_chave_aqui
  3. Rode:
       pip install requests python-dotenv
       python enrich_letterboxd.py
"""

import csv
import os
import time

import requests
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
TMDB_API_KEY  = os.getenv("TMDB_API_KEY")
RAW_DIR       = "letterboxd-jamillerochaghz-2026-03-30-21-47-utc"
OUTPUT_FILE   = "data/letterboxd_enriched.csv"
BASE_URL      = "https://api.themoviedb.org/3"
# ─────────────────────────────────────────────────────────────────────────────

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY não encontrada. Crie um arquivo .env com a chave.")


# ── Helpers: leitura dos CSVs do Letterboxd ───────────────────────────────────

def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_ratings_map(rows: list[dict]) -> dict:
    """Letterboxd URI → nota"""
    return {r["Letterboxd URI"]: r.get("Rating", "") for r in rows}


def build_diary_map(rows: list[dict]) -> dict:
    """Letterboxd URI → {watched_date, rewatch, tags}"""
    result = {}
    for r in rows:
        uri = r["Letterboxd URI"]
        # Pode haver múltiplas entradas do mesmo filme (rewatches); pega a mais recente
        if uri not in result:
            result[uri] = {
                "watched_date": r.get("Watched Date", ""),
                "rewatch":      r.get("Rewatch", ""),
                "tags":         r.get("Tags", ""),
            }
    return result


def build_reviews_set(rows: list[dict]) -> set:
    """Conjunto de URIs que têm review"""
    return {r["Letterboxd URI"] for r in rows}


def build_likes_set(rows: list[dict]) -> set:
    """Conjunto de URIs que você deu like"""
    return {r["Letterboxd URI"] for r in rows}


# ── Helpers: TMDB ─────────────────────────────────────────────────────────────

from typing import Optional

def search_movie(title: str, year: str) -> Optional[dict]:
    params = {"api_key": TMDB_API_KEY, "query": title, "language": "pt-BR"}
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
        "language": "pt-BR",
    }
    r = requests.get(f"{BASE_URL}/movie/{movie_id}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def empty_tmdb() -> dict:
    return {
        "tmdb_id":           "",
        "director":          "",
        "genre_primary":     "",
        "genre_secondary":   "",
        "genre_tertiary":    "",
        "country_primary":   "",
        "original_language": "",
        "runtime_min":       "",
        "vote_average":      "",
        "popularity":        "",
        "tagline":           "",
        "overview":          "",
    }


def fetch_tmdb(title: str, year: str) -> dict:
    result = search_movie(title, year)
    if not result:
        return empty_tmdb()

    details = get_details(result["id"])
    credits = details.get("credits", {})

    directors = [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"]
    genres    = [g["name"] for g in details.get("genres", [])]
    countries = [c["name"] for c in details.get("production_countries", [])]

    return {
        "tmdb_id":           details.get("id", ""),
        "director":          directors[0] if directors else "",
        "genre_primary":     genres[0] if len(genres) > 0 else "",
        "genre_secondary":   genres[1] if len(genres) > 1 else "",
        "genre_tertiary":    genres[2] if len(genres) > 2 else "",
        "country_primary":   countries[0] if countries else "",
        "original_language": details.get("original_language", ""),
        "runtime_min":       details.get("runtime", ""),
        "vote_average":      details.get("vote_average", ""),
        "popularity":        details.get("popularity", ""),
        "tagline":           details.get("tagline", ""),
        "overview":          details.get("overview", ""),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n📂 Lendo arquivos do Letterboxd...")

    watched  = read_csv(os.path.join(RAW_DIR, "watched.csv"))
    ratings  = build_ratings_map(read_csv(os.path.join(RAW_DIR, "ratings.csv")))
    diary    = build_diary_map(read_csv(os.path.join(RAW_DIR, "diary.csv")))
    reviews  = build_reviews_set(read_csv(os.path.join(RAW_DIR, "reviews.csv")))
    likes    = build_likes_set(read_csv(os.path.join(RAW_DIR, "likes", "films.csv")))

    total = len(watched)
    print(f"🎬 {total} filmes encontrados. Buscando no TMDB...\n")

    enriched = []
    for i, row in enumerate(watched, 1):
        title = row.get("Name", "").strip()
        year  = row.get("Year", "").strip()
        uri   = row.get("Letterboxd URI", "")

        print(f"[{i}/{total}]", end=" ")

        # Cruzamento interno
        diary_info = diary.get(uri, {})
        internal = {
            "rating":       ratings.get(uri, ""),
            "watched_date": diary_info.get("watched_date", ""),
            "rewatch":      diary_info.get("rewatch", ""),
            "tags":         diary_info.get("tags", ""),
            "has_review":   "Yes" if uri in reviews else "No",
            "liked":        "Yes" if uri in likes else "No",
        }

        # TMDB
        try:
            tmdb = fetch_tmdb(title, year)
            status = f"{tmdb['director']} | {tmdb['genre_primary']} | {tmdb['country_primary']}"
            print(f"✅ {title} → {status}")
        except Exception as e:
            print(f"❌ Erro em '{title}': {e}")
            tmdb = empty_tmdb()

        enriched.append({**row, **internal, **tmdb})
        time.sleep(0.25)

    # Salva
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fieldnames = list(enriched[0].keys())
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"\n✅ Pronto! Salvo em '{OUTPUT_FILE}'")
    print(f"   {len(enriched)} filmes | {len(fieldnames)} colunas")
    print(f"\n   Colunas finais:")
    for col in fieldnames:
        print(f"   · {col}")


if __name__ == "__main__":
    main()