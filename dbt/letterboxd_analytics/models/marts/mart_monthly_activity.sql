select
    profile_id,

    date_trunc(
        'month',
        watched_date
    )::date as month,

    count(*) as watch_events,

    count(distinct tmdb_id) as unique_movies,

    sum(
        coalesce(runtime_min, 0)
    ) as minutes_watched,

    round(
        sum(coalesce(runtime_min, 0)) / 60.0,
        2
    ) as hours_watched,

    count(*) filter (
        where lower(coalesce(rewatch, '')) in ('yes', 'true')
    ) as rewatches,

    round(
        avg(rating),
        2
    ) as average_rating

from {{ ref('fct_watch_events') }}

where watched_date is not null

group by
    profile_id,
    date_trunc('month', watched_date)