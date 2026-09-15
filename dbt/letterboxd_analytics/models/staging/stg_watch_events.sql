select
    watch_event_id,
    profile_id,
    letterboxd_uri,
    title,
    nullif(release_year, '')::integer as release_year,
    watched_date::date as watched_date,
    rewatch,
    tags

from {{ source('raw', 'watch_events') }}