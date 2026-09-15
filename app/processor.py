import tempfile
import zipfile
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import pandas as pd

from src.letterboxd_pipeline.letterboxd import read_csv_file

MAX_ZIP_SIZE_MB = 50
MAX_UNCOMPRESSED_SIZE_MB = 200
MAX_FILES = 100


def safe_extract_zip(uploaded_file, destination: Path) -> None:
    """
    Safely extract a Letterboxd ZIP export.

    Protects against:
    - ZIP path traversal
    - oversized uploads
    - ZIP bombs
    - excessive file counts
    """

    destination = destination.resolve()

    # -----------------------------------------------------
    # COMPRESSED FILE SIZE
    # -----------------------------------------------------

    uploaded_file.seek(0, 2)
    zip_size = uploaded_file.tell()
    uploaded_file.seek(0)

    max_zip_bytes = MAX_ZIP_SIZE_MB * 1024 * 1024

    if zip_size > max_zip_bytes:
        raise ValueError(
            f"ZIP file is too large. "
            f"Maximum allowed size is {MAX_ZIP_SIZE_MB} MB."
        )

    # -----------------------------------------------------
    # OPEN ZIP
    # -----------------------------------------------------

    try:
        archive = ZipFile(uploaded_file)

    except BadZipFile:
        raise ValueError(
            "The uploaded file is not a valid ZIP archive."
        )

    with archive:

        files = archive.infolist()

        # -------------------------------------------------
        # FILE COUNT
        # -------------------------------------------------

        if len(files) > MAX_FILES:
            raise ValueError(
                f"The ZIP contains too many files. "
                f"Maximum allowed is {MAX_FILES}."
            )

        # -------------------------------------------------
        # UNCOMPRESSED SIZE
        # -------------------------------------------------

        total_uncompressed_size = sum(
            file.file_size
            for file in files
        )

        max_uncompressed_bytes = (
            MAX_UNCOMPRESSED_SIZE_MB
            * 1024
            * 1024
        )

        if total_uncompressed_size > max_uncompressed_bytes:
            raise ValueError(
                "The extracted files are too large. "
                f"Maximum allowed size is "
                f"{MAX_UNCOMPRESSED_SIZE_MB} MB."
            )

        # -------------------------------------------------
        # VALIDATE EACH PATH
        # -------------------------------------------------

        for file in files:

            file_path = (
                destination
                / file.filename
            ).resolve()

            try:
                file_path.relative_to(
                    destination
                )

            except ValueError:
                raise ValueError(
                    "Unsafe path detected inside ZIP file."
                )

        # -------------------------------------------------
        # EXTRACT
        # -------------------------------------------------

        archive.extractall(
            destination
        )

def extract_letterboxd_zip(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    zip_path = Path(temp_dir.name) / "letterboxd.zip"

    with open(zip_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    safe_extract_zip(
    uploaded_file,
    Path(temp_dir), 
    )

    return temp_dir


def find_export_directory(temp_dir):
    root = Path(temp_dir.name)
    matches = list(root.rglob("watched.csv"))

    if not matches:
        raise ValueError(
            "This ZIP does not appear to contain a valid Letterboxd export."
        )

    return matches[0].parent


def load_letterboxd_data(export_dir):
    export_dir = Path(export_dir)

    files = {
        "watched": export_dir / "watched.csv",
        "ratings": export_dir / "ratings.csv",
        "diary": export_dir / "diary.csv",
        "reviews": export_dir / "reviews.csv",
        "watchlist": export_dir / "watchlist.csv",
    }

    data = {}

    for name, path in files.items():
        data[name] = pd.DataFrame(read_csv_file(path))

    likes_path = export_dir / "likes" / "films.csv"
    data["likes"] = pd.DataFrame(read_csv_file(likes_path))

    return data