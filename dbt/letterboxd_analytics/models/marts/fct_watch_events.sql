select
    w.watch_event_id,
    w.profile_id,
    w.letterboxd_uri,

    u.tmdb_id,

    coalesce(m.title, w.title) as title,
    coalesce(m.release_year, w.release_year) as release_year,

    w.watched_date,
    w.rewatch,
    w.tags,

    u.rating,
    u.liked,
    u.has_review,

    m.director,
    m.genre_primary,
    m.genre_secondary,
    m.country_primary,
    m.original_language,
    m.runtime_min,
    m.vote_average

from {{ ref('stg_watch_events') }} w

left join {{ ref('stg_user_movies') }} u
    on w.profile_id = u.profile_id
    and w.letterboxd_uri = u.letterboxd_uri

left join {{ ref('stg_movies') }} m
    on u.tmdb_id = m.tmdb_id