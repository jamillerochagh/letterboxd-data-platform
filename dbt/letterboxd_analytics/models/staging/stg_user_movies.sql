select
    profile_id,
    tmdb_id::bigint as tmdb_id,
    letterboxd_uri,
    rating::numeric as rating,
    liked,
    has_review,
    tags

from {{ source('raw', 'user_movies') }}