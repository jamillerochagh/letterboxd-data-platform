import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd

from src.letterboxd_pipeline.letterboxd import read_csv_file


# =========================================================
# UPLOAD SECURITY LIMITS
# =========================================================

MAX_ZIP_SIZE_MB = 50
MAX_UNCOMPRESSED_SIZE_MB = 200
MAX_FILES = 100


# =========================================================
# SAFE ZIP EXTRACTION
# =========================================================

def safe_extract_zip(
    uploaded_file,
    destination: Path,
) -> None:
    """
    Safely extract a Letterboxd ZIP export.

    Protects against:
    - invalid ZIP files
    - oversized uploads
    - excessive file counts
    - ZIP bombs
    - ZIP path traversal
    """

    destination = Path(destination).resolve()

    # Make sure the destination exists.
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # COMPRESSED FILE SIZE
    # -----------------------------------------------------

    uploaded_file.seek(0, 2)

    zip_size = uploaded_file.tell()

    uploaded_file.seek(0)

    max_zip_bytes = (
        MAX_ZIP_SIZE_MB
        * 1024
        * 1024
    )

    if zip_size > max_zip_bytes:

        raise ValueError(
            f"ZIP file is too large. "
            f"Maximum allowed size is "
            f"{MAX_ZIP_SIZE_MB} MB."
        )

    # -----------------------------------------------------
    # OPEN ZIP
    # -----------------------------------------------------

    try:

        archive = ZipFile(
            uploaded_file
        )

    except BadZipFile as exc:

        raise ValueError(
            "The uploaded file is not a valid ZIP archive."
        ) from exc

    # -----------------------------------------------------
    # VALIDATE ZIP
    # -----------------------------------------------------

    with archive:

        members = archive.infolist()

        # -------------------------------------------------
        # FILE COUNT
        # -------------------------------------------------

        file_members = [
            member
            for member in members
            if not member.is_dir()
        ]

        if len(file_members) > MAX_FILES:

            raise ValueError(
                f"The ZIP contains too many files. "
                f"Maximum allowed is "
                f"{MAX_FILES}."
            )

        # -------------------------------------------------
        # TOTAL UNCOMPRESSED SIZE
        # -------------------------------------------------

        total_uncompressed_size = sum(
            member.file_size
            for member in file_members
        )

        max_uncompressed_bytes = (
            MAX_UNCOMPRESSED_SIZE_MB
            * 1024
            * 1024
        )

        if (
            total_uncompressed_size
            > max_uncompressed_bytes
        ):

            raise ValueError(
                "The extracted files are too large. "
                f"Maximum allowed size is "
                f"{MAX_UNCOMPRESSED_SIZE_MB} MB."
            )

        # -------------------------------------------------
        # VALIDATE EVERY PATH
        # -------------------------------------------------

        for member in members:

            member_path = (
                destination
                / member.filename
            ).resolve()

            try:

                member_path.relative_to(
                    destination
                )

            except ValueError as exc:

                raise ValueError(
                    "Unsafe path detected inside ZIP file."
                ) from exc

        # -------------------------------------------------
        # EXTRACT
        # -------------------------------------------------

        archive.extractall(
            destination
        )


# =========================================================
# EXTRACT LETTERBOXD EXPORT
# =========================================================

def extract_letterboxd_zip(
    uploaded_file,
):
    """
    Extract an uploaded Letterboxd ZIP into
    a temporary directory.

    Returns the TemporaryDirectory object so
    the directory remains alive while the app
    processes the export.
    """

    temp_dir = (
        tempfile.TemporaryDirectory()
    )

    destination = Path(
        temp_dir.name
    )

    try:

        safe_extract_zip(
            uploaded_file,
            destination,
        )

    except Exception:

        # Clean up the temporary directory
        # if extraction fails.

        temp_dir.cleanup()

        raise

    return temp_dir


# =========================================================
# FIND LETTERBOXD EXPORT DIRECTORY
# =========================================================

def find_export_directory(
    temp_dir,
):
    """
    Locate watched.csv inside the extracted ZIP.

    Letterboxd exports may contain their CSV files
    directly or inside another directory.
    """

    root = Path(
        temp_dir.name
    )

    matches = list(
        root.rglob(
            "watched.csv"
        )
    )

    if not matches:

        raise ValueError(
            "This ZIP does not appear to contain "
            "a valid Letterboxd export."
        )

    return matches[0].parent


# =========================================================
# SAFE CSV LOADER
# =========================================================

def load_optional_csv(
    path: Path,
) -> pd.DataFrame:
    """
    Load a Letterboxd CSV when available.

    Some exports may not contain every optional
    file, so missing files return an empty
    DataFrame instead of crashing the app.
    """

    path = Path(path)

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.DataFrame(
            read_csv_file(
                path
            )
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to read {path.name}."
        ) from exc


# =========================================================
# LOAD LETTERBOXD DATA
# =========================================================

def load_letterboxd_data(
    export_dir,
):
    """
    Load the supported files from a Letterboxd export.
    """

    export_dir = Path(
        export_dir
    )

    # watched.csv is required because it identifies
    # a valid Letterboxd export.

    watched_path = (
        export_dir
        / "watched.csv"
    )

    if not watched_path.exists():

        raise ValueError(
            "watched.csv was not found "
            "in the Letterboxd export."
        )

    files = {
        "watched":
            watched_path,

        "ratings":
            export_dir
            / "ratings.csv",

        "diary":
            export_dir
            / "diary.csv",

        "reviews":
            export_dir
            / "reviews.csv",

        "watchlist":
            export_dir
            / "watchlist.csv",
    }

    data = {}

    for name, path in files.items():

        data[name] = (
            load_optional_csv(
                path
            )
        )

    # -----------------------------------------------------
    # LIKES
    # -----------------------------------------------------

    likes_path = (
        export_dir
        / "likes"
        / "films.csv"
    )

    data["likes"] = (
        load_optional_csv(
            likes_path
        )
    )

    return data