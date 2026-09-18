import json
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from .database import get_engine

def _json_safe(value):
    """
    Recursively convert pandas/numpy objects into values
    that can safely be serialized as JSON.
    """
    import numpy as np

    if isinstance(value, dict):
        safe_dict = {}

        for key, item in value.items():
            # JSON object keys must be native strings.
            safe_key = str(key)

            safe_dict[safe_key] = _json_safe(item)

        return safe_dict

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if value is pd.NA:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value

def dataframe_to_records(
    df: pd.DataFrame,
) -> list[dict]:
    """Convert a DataFrame into JSON-safe records."""

    if df is None or df.empty:
        return []

    records = df.to_dict(
        orient="records"
    )

    return _json_safe(records)

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

    safe_taste_profile = _json_safe(
        taste_profile
    )

    safe_movie_records = _json_safe(
        movie_records
    )

    profile_json = json.dumps(
        safe_taste_profile
    )

    history_json = json.dumps(
        safe_movie_records
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

# =========================================================
# PERSISTED BLEND RESULTS
# =========================================================

def ensure_blend_results_table() -> bool:
    """
    Create the persisted Blend result cache if it does not exist.

    The table stores derived recommendation rows only. Original ZIP files
    are never stored here.
    """
    query = text(
        """
        CREATE TABLE IF NOT EXISTS public.blend_results (
            blend_id UUID PRIMARY KEY
                REFERENCES public.blends(blend_id)
                ON DELETE CASCADE,
            recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )

    try:
        engine = get_engine()

        with engine.begin() as connection:
            connection.execute(query)

        return True

    except Exception as error:
        print(
            "BLEND RESULT TABLE ERROR | "
            f"{type(error).__name__}: {error}"
        )
        return False


def get_blend_recommendations(
    blend_id: str,
) -> pd.DataFrame | None:
    """
    Return persisted shared recommendations for a Blend.

    None means no persisted result exists yet.
    """
    if not ensure_blend_results_table():
        return None

    query = text(
        """
        SELECT recommendations
        FROM public.blend_results
        WHERE blend_id = :blend_id
        LIMIT 1
        """
    )

    try:
        engine = get_engine()

        with engine.connect() as connection:
            value = connection.execute(
                query,
                {"blend_id": blend_id},
            ).scalar_one_or_none()

        if value is None:
            return None

        if isinstance(value, str):
            value = json.loads(value)

        return pd.DataFrame(
            value or []
        )

    except Exception as error:
        print(
            "BLEND RESULT READ ERROR | "
            f"{type(error).__name__}: {error}"
        )
        return None


def save_blend_recommendations(
    blend_id: str,
    recommendations: pd.DataFrame,
) -> bool:
    """Persist derived shared recommendations for fast future opens."""
    if not ensure_blend_results_table():
        return False

    records = dataframe_to_records(
        recommendations
    )

    payload = json.dumps(
        _json_safe(records)
    )

    query = text(
        """
        INSERT INTO public.blend_results (
            blend_id,
            recommendations,
            updated_at
        )
        VALUES (
            :blend_id,
            CAST(:recommendations AS JSONB),
            NOW()
        )
        ON CONFLICT (blend_id)
        DO UPDATE SET
            recommendations = EXCLUDED.recommendations,
            updated_at = NOW()
        """
    )

    try:
        engine = get_engine()

        with engine.begin() as connection:
            connection.execute(
                query,
                {
                    "blend_id": blend_id,
                    "recommendations": payload,
                },
            )

        print(
            "BLEND RESULT SAVED | "
            f"blend_id={blend_id} | "
            f"recommendations={len(records)}"
        )
        return True

    except Exception as error:
        print(
            "BLEND RESULT WRITE ERROR | "
            f"{type(error).__name__}: {error}"
        )
        return False
