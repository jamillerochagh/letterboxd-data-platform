select
    tmdb_id::bigint as tmdb_id,
    title,
    release_year::integer as release_year,
    director,
    genre_primary,
    genre_secondary,
    genre_tertiary,
    country_primary,
    original_language,
    runtime_min::integer as runtime_min,
    vote_average::numeric as vote_average,
    popularity::numeric as popularity,
    tagline,
    overview

from {{ source('raw', 'movies') }}

where tmdb_id is not null