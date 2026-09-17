import json
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from .database import get_engine


def _json_safe(value):
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into JSON-safe records."""
    if df is None or df.empty:
        return []

    clean = df.copy()

    for column in clean.columns:
        clean[column] = clean[column].map(_json_safe)

    return clean.to_dict(orient="records")


def create_blend(creator_name: str | None = None) -> str:
    """
    Create a new Movie Blend and return its public UUID.
    """
    blend_id = str(uuid.uuid4())

    query = text(
        """
        INSERT INTO public.blends (
            blend_id,
            creator_name,
            creator_ready,
            friend_ready
        )
        VALUES (
            :blend_id,
            :creator_name,
            FALSE,
            FALSE
        )
        """
    )

    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "blend_id": blend_id,
                "creator_name": creator_name,
            },
        )

    print(f"BLEND CREATED | blend_id={blend_id}")

    return blend_id


def get_blend(blend_id: str) -> dict | None:
    """
    Retrieve an active Blend.
    Expired or unknown Blend IDs return None.
    """
    query = text(
        """
        SELECT
            blend_id,
            created_at,
            expires_at,
            creator_name,
            friend_name,
            creator_ready,
            friend_ready
        FROM public.blends
        WHERE blend_id = :blend_id
          AND expires_at > NOW()
        LIMIT 1
        """
    )

    engine = get_engine()

    with engine.connect() as connection:
        row = connection.execute(
            query,
            {"blend_id": blend_id},
        ).mappings().first()

    if not row:
        return None

    return dict(row)


def save_blend_profile(
    blend_id: str,
    profile_slot: str,
    display_name: str,
    taste_profile: dict,
    movie_history: pd.DataFrame,
) -> None:
    """
    Save or replace one side of a Blend.

    profile_slot must be either 'creator' or 'friend'.
    """
    if profile_slot not in {"creator", "friend"}:
        raise ValueError(
            "profile_slot must be 'creator' or 'friend'"
        )

    blend = get_blend(blend_id)

    if not blend:
        raise ValueError("Blend does not exist or has expired.")

    movie_records = dataframe_to_records(movie_history)

    profile_json = json.dumps(
        taste_profile,
        default=_json_safe,
    )

    history_json = json.dumps(
        movie_records,
        default=_json_safe,
    )

    profile_query = text(
        """
        INSERT INTO public.blend_profiles (
            blend_id,
            profile_slot,
            display_name,
            taste_profile,
            movie_history
        )
        VALUES (
            :blend_id,
            :profile_slot,
            :display_name,
            CAST(:taste_profile AS JSONB),
            CAST(:movie_history AS JSONB)
        )
        ON CONFLICT (blend_id, profile_slot)
        DO UPDATE SET
            display_name = EXCLUDED.display_name,
            taste_profile = EXCLUDED.taste_profile,
            movie_history = EXCLUDED.movie_history,
            created_at = NOW()
        """
    )

    if profile_slot == "creator":
        status_query = text(
            """
            UPDATE public.blends
            SET
                creator_name = :display_name,
                creator_ready = TRUE
            WHERE blend_id = :blend_id
            """
        )
    else:
        status_query = text(
            """
            UPDATE public.blends
            SET
                friend_name = :display_name,
                friend_ready = TRUE
            WHERE blend_id = :blend_id
            """
        )

    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(
            profile_query,
            {
                "blend_id": blend_id,
                "profile_slot": profile_slot,
                "display_name": display_name,
                "taste_profile": profile_json,
                "movie_history": history_json,
            },
        )

        connection.execute(
            status_query,
            {
                "blend_id": blend_id,
                "display_name": display_name,
            },
        )

    print(
        "BLEND PROFILE SAVED | "
        f"blend_id={blend_id} | "
        f"slot={profile_slot}"
    )


def get_blend_profile(
    blend_id: str,
    profile_slot: str,
) -> dict | None:
    """Retrieve one processed profile from a Blend."""
    if profile_slot not in {"creator", "friend"}:
        raise ValueError(
            "profile_slot must be 'creator' or 'friend'"
        )

    query = text(
        """
        SELECT
            display_name,
            taste_profile,
            movie_history,
            created_at
        FROM public.blend_profiles
        WHERE blend_id = :blend_id
          AND profile_slot = :profile_slot
        LIMIT 1
        """
    )

    engine = get_engine()

    with engine.connect() as connection:
        row = connection.execute(
            query,
            {
                "blend_id": blend_id,
                "profile_slot": profile_slot,
            },
        ).mappings().first()

    if not row:
        return None

    result = dict(row)

    if isinstance(result["taste_profile"], str):
        result["taste_profile"] = json.loads(
            result["taste_profile"]
        )

    if isinstance(result["movie_history"], str):
        result["movie_history"] = json.loads(
            result["movie_history"]
        )

    return result


def get_blend_profiles(blend_id: str) -> dict:
    """Retrieve both sides of a Blend."""
    return {
        "creator": get_blend_profile(
            blend_id,
            "creator",
        ),
        "friend": get_blend_profile(
            blend_id,
            "friend",
        ),
    }


def blend_is_ready(blend_id: str) -> bool:
    """Return True once both users have completed their profiles."""
    blend = get_blend(blend_id)

    if not blend:
        return False

    return bool(
        blend["creator_ready"]
        and blend["friend_ready"]
    )


def delete_expired_blends() -> int:
    """Delete expired Blends and their profiles."""
    query = text(
        """
        DELETE FROM public.blends
        WHERE expires_at <= NOW()
        """
    )

    engine = get_engine()

    with engine.begin() as connection:
        result = connection.execute(query)

    deleted = result.rowcount or 0

    if deleted:
        print(
            f"BLEND CLEANUP | deleted={deleted}"
        )

    return deleted