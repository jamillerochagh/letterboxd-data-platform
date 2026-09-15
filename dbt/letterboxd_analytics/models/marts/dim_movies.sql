select
    tmdb_id,
    title,
    release_year,
    director,
    genre_primary,
    genre_secondary,
    genre_tertiary,
    country_primary,
    original_language,
    runtime_min,
    vote_average,
    popularity,
    tagline,
    overview

from {{ ref('stg_movies') }}