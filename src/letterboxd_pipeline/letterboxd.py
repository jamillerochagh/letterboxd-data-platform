import csv
from pathlib import Path


def read_csv_file(path: Path) -> list[dict]:
    """Read a CSV file and return its rows as dictionaries."""
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_ratings_map(rows: list[dict]) -> dict:
    """Map Letterboxd URI to the user's rating."""
    return {
        row["Letterboxd URI"]: row.get("Rating", "")
        for row in rows
    }


def build_diary_map(rows: list[dict]) -> dict:
    """Map Letterboxd URI to diary information."""
    result = {}

    for row in rows:
        uri = row["Letterboxd URI"]

        if uri not in result:
            result[uri] = {
                "watched_date": row.get("Watched Date", ""),
                "rewatch": row.get("Rewatch", ""),
                "tags": row.get("Tags", ""),
            }

    return result


def build_uri_set(rows: list[dict]) -> set:
    """Return the set of Letterboxd URIs contained in a dataset."""
    return {
        row["Letterboxd URI"]
        for row in rows
        if row.get("Letterboxd URI")
    }


def load_letterboxd_export(export_dir: str | Path) -> dict:
    """
    Load the relevant datasets from an extracted Letterboxd export.

    The function works with any user's Letterboxd export.
    """
    export_dir = Path(export_dir)

    watched_path = export_dir / "watched.csv"

    if not watched_path.exists():
        raise FileNotFoundError(
            f"watched.csv was not found in: {export_dir}"
        )

    likes_path = export_dir / "likes" / "films.csv"

    return {
        "watched": read_csv_file(watched_path),
        "ratings": build_ratings_map(
            read_csv_file(export_dir / "ratings.csv")
        ),
        "diary": build_diary_map(
            read_csv_file(export_dir / "diary.csv")
        ),
        "reviews": build_uri_set(
            read_csv_file(export_dir / "reviews.csv")
        ),
        "likes": build_uri_set(
            read_csv_file(likes_path)
        ),
    }