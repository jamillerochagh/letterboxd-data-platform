select
    u.profile_id,
    u.tmdb_id,
    u.letterboxd_uri,

    m.title,
    m.release_year,
    m.director,
    m.genre_primary,
    m.genre_secondary,
    m.genre_tertiary,
    m.country_primary,
    m.original_language,
    m.runtime_min,
    m.vote_average,
    m.popularity,

    u.rating,
    u.liked,
    u.has_review,
    u.tags,

    count(w.watch_event_id) as diary_watch_count,

    min(w.watched_date) as first_diary_watch_date,
    max(w.watched_date) as latest_diary_watch_date,

    case
        when count(w.watch_event_id) > 1 then true
        else false
    end as has_rewatch

from {{ ref('stg_user_movies') }} u

left join {{ ref('stg_movies') }} m
    on u.tmdb_id = m.tmdb_id

left join {{ ref('stg_watch_events') }} w
    on u.profile_id = w.profile_id
    and u.letterboxd_uri = w.letterboxd_uri

group by
    u.profile_id,
    u.tmdb_id,
    u.letterboxd_uri,
    m.title,
    m.release_year,
    m.director,
    m.genre_primary,
    m.genre_secondary,
    m.genre_tertiary,
    m.country_primary,
    m.original_language,
    m.runtime_min,
    m.vote_average,
    m.popularity,
    u.rating,
    u.liked,
    u.has_review,
    u.tags