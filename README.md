# Letterboxd Data Platform

An end-to-end data platform that transforms a personal Letterboxd export
into an enriched analytics experience, personalized movie
recommendations, and collaborative taste matching.

The project combines data ingestion, external API enrichment, PostgreSQL
caching, analytics, recommendation logic, and a Streamlit application in
a single production-oriented pipeline.

**Live application:** https://letterboxd-data-platform.streamlit.app\
**Repository:**
https://github.com/jamillerochagh/letterboxd-data-platform

------------------------------------------------------------------------

## Overview

Letterboxd exports contain useful viewing history, but the raw files are
limited for deeper analysis: they do not include rich movie metadata
such as genres, directors, cast, countries, runtime, popularity,
posters, or broader audience signals.

This project builds a reusable data pipeline around that export.

A user uploads a Letterboxd ZIP and the application:

1.  extracts and loads the relevant Letterboxd datasets;
2.  normalizes ratings, diary activity, likes, and movie identifiers;
3.  checks PostgreSQL for previously enriched movies;
4.  enriches cache misses through the TMDB API;
5.  writes new metadata back to the shared database cache;
6.  builds a taste profile from the enriched history;
7.  generates analytics and personalized recommendations;
8.  optionally creates a temporary Movie Blend with another Letterboxd
    user.

The result is not only a dashboard, but a complete data product with
ingestion, enrichment, persistence, transformation, recommendation, and
presentation layers.

------------------------------------------------------------------------

## Architecture

``` text
                         LETTERBOXD EXPORT
                              ZIP / CSV
                                 |
                                 v
                    +--------------------------+
                    |     Ingestion Layer      |
                    | extract + load + clean   |
                    +--------------------------+
                                 |
                                 v
                    +--------------------------+
                    |   Identity / Matching    |
                    | URI + title + year       |
                    +--------------------------+
                                 |
                                 v
                 +--------------------------------+
                 |       Enrichment Layer         |
                 |                                |
                 | PostgreSQL cache lookup        |
                 |          |                     |
                 |     cache miss                 |
                 |          v                     |
                 |      TMDB API                  |
                 |  parallel enrichment           |
                 +--------------------------------+
                         |              |
                         |              v
                         |      +----------------+
                         +----> | PostgreSQL /   |
                                | Supabase Cache |
                                +----------------+
                                      |
                                      v
                         +------------------------+
                         | Analytics & Taste      |
                         | Profile Layer          |
                         +------------------------+
                            |                |
                            v                v
                   +---------------+   +------------------+
                   | Personalized  |   |   Movie Blend    |
                   | Recommender   |   | shared profiles  |
                   +---------------+   +------------------+
                            \                /
                             \              /
                              v            v
                           +------------------+
                           |    Streamlit     |
                           |  Data Product UI |
                           +------------------+
```

### Data flow

``` text
Letterboxd ZIP
    -> pandas ingestion
    -> normalization and movie matching
    -> selective PostgreSQL cache lookup
    -> concurrent TMDB enrichment for cache misses
    -> batch PostgreSQL upsert
    -> enriched viewing history
    -> taste profile
    -> analytics / recommendations / Movie Blend
    -> Streamlit
```

------------------------------------------------------------------------

## Core Features

### Personal Analytics

The Overview experience turns viewing history into a Letterboxd-style
personal retrospective.

It includes:

-   total movies and viewing activity;
-   average ratings and rating distribution;
-   first and most recent watches;
-   most watched genres;
-   most watched directors;
-   country-level cinema map;
-   exact five-star movies;
-   lowest-rated movies;
-   frequently watched actors;
-   year-based filtering using diary watch dates.

The analytics layer combines Letterboxd activity with enriched TMDB
metadata rather than relying only on the original export.

### You vs Crowd

This section compares the user's ratings with broader movie-audience
signals.

It is designed to surface where personal taste aligns with or diverges
from general reception while keeping the user's own rating history as
the primary signal.

### Personalized Recommendations

The recommendation engine converts viewing history into a taste profile
and discovers new candidate movies through TMDB.

Recommendations are grouped into interpretable sections such as:

-   Top Picks;
-   From Directors You Love;
-   Hidden Gems;
-   Outside Your Comfort Zone;
-   Safe Bets.

Each recommendation receives a **Taste Score from 0 to 100**. The score
is a ranking signal, not a probability.

Candidate generation uses broad genre discovery followed by preference
scoring, quality signals, filtering, and diversification.

### Movie Blend

Movie Blend creates a temporary shared recommendation experience between
two Letterboxd users.

The creator generates an invitation link. The second user opens the link
and uploads their own Letterboxd export. The raw ZIP files are not
persisted; the application stores only derived profile/history
information required for the Blend.

The shared result includes:

-   Taste Match;
-   movies in common;
-   average rating gap;
-   shared top genre;
-   movies both users rated exactly 5 stars;
-   biggest rating disagreements;
-   shared movie recommendations;
-   individual match scores;
-   combined Blend Score.

Blend recommendations are generated from both taste profiles and use
consensus-oriented scoring so that a movie must work reasonably well for
both users rather than only one.

Blend sessions use UUID-based invitation links and expire after 30 days.

------------------------------------------------------------------------

## Data Engineering Highlights

### Selective PostgreSQL Cache

TMDB metadata is cached in PostgreSQL so the application does not
repeatedly enrich movies that have already been processed.

Instead of downloading the entire shared movie cache for every upload,
the application queries only titles relevant to the current Letterboxd
history.

This reduces:

-   database transfer;
-   application memory usage;
-   startup latency;
-   unnecessary TMDB requests.

### Concurrent API Enrichment

Movies missing from the database cache are enriched concurrently through
TMDB.

The enrichment layer:

1.  identifies unique cache misses;
2.  processes requests using a bounded thread pool;
3.  reports progress to the application;
4.  combines successful results;
5.  writes them to PostgreSQL in a batch.

This replaced sequential enrichment and substantially reduced processing
time for large Letterboxd histories.

### Batch Upserts

New TMDB metadata is written back to `raw.movies` using PostgreSQL batch
upserts keyed by `tmdb_id`.

Before persistence, database-bound values are normalized so missing
numeric values become SQL `NULL` rather than invalid empty strings.

The cache is intentionally non-critical: a database read or write
failure is logged without preventing the user from completing the
analysis through TMDB.

### Streamlit Rerun Optimization

Streamlit reruns the application when widgets change.

To avoid repeating expensive work on every interaction, the full
analysis bundle is stored in session state and keyed by a fingerprint of
the uploaded ZIP.

The cached bundle contains the processed Letterboxd datasets, enriched
history, preference data, and taste profile.

As a result, switching sections or interacting with Movie Blend does not
trigger the complete ingestion and enrichment pipeline again.

### Persistent Blend Results

Movie Blend profiles and recommendation results are persisted in
PostgreSQL.

This separates the expensive recommendation computation from
presentation: once a Blend result has been generated, both participants
can retrieve the saved result rather than rebuilding it on every
application rerun.

------------------------------------------------------------------------

## Recommendation Approach

The recommender is intentionally hybrid and interpretable rather than a
single black-box model.

Signals include:

-   preferred genres;
-   director affinity;
-   user rating behavior;
-   movie quality signals;
-   TMDB popularity and vote data;
-   candidate relevance;
-   novelty;
-   diversification.

The system first creates a user preference profile from historical
behavior. It then retrieves a broader candidate pool, enriches the
candidates, scores them against the profile, applies quality
adjustments, and diversifies the final output.

For Movie Blend, the recommendation layer combines both users' match
scores with additional weight on the weaker individual match. This
discourages recommendations that are excellent for one participant but
poor for the other.

------------------------------------------------------------------------

## Performance Design

Several optimizations were introduced after profiling the deployed
application:

  Area                    Optimization
  ----------------------- -----------------------------------------------------------
  TMDB enrichment         Concurrent requests with bounded workers
  Database reads          Query only movies relevant to the current upload
  Database writes         Batch upserts
  Duplicate movies        Deduplicated before enrichment
  Streamlit reruns        Full analysis bundle stored in session state
  Recommendations         Lazy computation when the section is opened
  Movie Blend             Explicit creator/friend flow and persistent profiles
  Blend recommendations   Persistent derived results
  Posters                 Metadata reused instead of repeated render-time API calls
  Failure handling        Database cache failures remain non-fatal

The goal is to minimize repeated I/O and external API work while keeping
the application responsive.

------------------------------------------------------------------------

## Technology Stack

  Layer                       Technology
  --------------------------- ---------------------------
  Language                    Python
  Data processing             pandas
  Application                 Streamlit
  Database                    PostgreSQL
  Production database         Supabase
  ORM / database access       SQLAlchemy
  External metadata           TMDB API
  Recommendation logic        Python / pandas
  Local database              PostgreSQL with Docker
  Transform experimentation   dbt project structure
  Deployment                  Streamlit Community Cloud
  Version control             Git / GitHub

------------------------------------------------------------------------

## Project Structure

``` text
letterboxd-data-platform/
|
|-- app/
|   |-- app.py
|   |-- enrichment.py
|   |-- processor.py
|   `-- blend_ui.py
|
|-- src/
|   `-- letterboxd_pipeline/
|       |-- config.py
|       |-- database.py
|       |-- enrichment.py
|       |-- letterboxd.py
|       |-- models.py
|       |-- recommendations.py
|       |-- tmdb.py
|       |-- blend.py
|       `-- blend_engine.py
|
|-- dbt/
|   `-- ...
|
|-- .streamlit/
|   `-- config.toml
|
|-- requirements.txt
|-- .gitignore
`-- README.md
```

The `app` package contains the Streamlit-facing orchestration and
presentation logic. Reusable pipeline, persistence, enrichment, and
recommendation logic live under `src/letterboxd_pipeline`.

------------------------------------------------------------------------

## Data Model

### `raw.movies`

Shared TMDB enrichment cache.

Key fields include:

``` text
tmdb_id
title
release_year
director
cast_top
genre_primary
genre_secondary
genre_tertiary
country_primary
original_language
runtime_min
vote_average
popularity
tagline
overview
poster_path
```

`tmdb_id` is used as the conflict key for cache upserts when available.

### `public.blends`

Stores temporary Movie Blend sessions.

``` text
blend_id
created_at
expires_at
creator_name
friend_name
creator_ready
friend_ready
```

### `public.blend_profiles`

Stores derived participant data for a Blend.

``` text
id
blend_id
profile_slot
display_name
taste_profile
movie_history
created_at
```

A Blend contains one `creator` profile and one `friend` profile.

### `public.blend_results`

Stores derived shared recommendation results so they can be reused
across application reruns and by both Blend participants.

------------------------------------------------------------------------

## Local Setup

### 1. Clone the repository

``` bash
git clone https://github.com/jamillerochagh/letterboxd-data-platform.git
cd letterboxd-data-platform
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

``` bash
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file:

``` text
DATABASE_URL=your_postgresql_connection_string
TMDB_API_KEY=your_tmdb_api_key
```

Do not commit `.env` or production secrets.

### 5. Run the application

``` bash
streamlit run app/app.py
```

------------------------------------------------------------------------

## Production Deployment

The public application is deployed on Streamlit Community Cloud and uses
Supabase PostgreSQL for shared persistence.

Production credentials are supplied through Streamlit Secrets rather
than committed files.

Required secrets:

``` text
DATABASE_URL
TMDB_API_KEY
```

The application is designed so that PostgreSQL acts primarily as an
optimization and persistence layer. TMDB enrichment can continue when
movie-cache operations fail, preventing a temporary cache issue from
making the core analysis unavailable.

------------------------------------------------------------------------

## Privacy and Security

The application processes user-provided Letterboxd exports.

Important design choices:

-   uploaded ZIP files are processed for the active analysis and are not
    intentionally persisted as raw files;
-   Movie Blend stores derived profile/history data rather than the
    uploaded ZIP;
-   Blend invitation identifiers are UUIDs;
-   Blend records expire after 30 days;
-   database credentials and TMDB keys are supplied through environment
    variables or Streamlit Secrets;
-   `.env`, local secrets, exports, and raw user data are excluded from
    version control.

Movie Blend invitation links should be treated as bearer links: anyone
with a valid active link may be able to access that Blend.

------------------------------------------------------------------------

## Engineering Decisions

### Why PostgreSQL instead of enriching everything on every upload?

TMDB metadata is largely reusable across users. Persisting enriched
movies converts repeated API work into inexpensive database lookups and
allows the application to improve its shared cache over time.

### Why keep database failures non-fatal?

The movie cache is an optimization. A cache outage should make
processing slower, not make the application unusable.

### Why cache the full analysis in session state?

Streamlit's execution model reruns the script after widget interactions.
Without session-level caching, even a simple navigation action could
repeat ZIP parsing, joins, enrichment checks, and taste-profile
generation.

### Why persist Movie Blend recommendations?

The Blend recommendation step is more expensive than rendering the
comparison itself. Persisting the derived result avoids recomputing the
same recommendations for both participants and across reruns.

### Why not store the uploaded ZIP for Movie Blend?

The raw export is unnecessary once the required taste profile and movie
history have been derived. Avoiding raw ZIP persistence reduces
unnecessary retention of user data.

------------------------------------------------------------------------

## Current Scope

The current release focuses on a production-ready personal analytics and
recommendation workflow.

Implemented:

-   Letterboxd ZIP ingestion;
-   ratings, diary, likes, and viewing-history processing;
-   TMDB enrichment;
-   PostgreSQL shared cache;
-   selective cache reads;
-   concurrent enrichment;
-   batch upserts;
-   personal analytics;
-   geographic cinema analysis;
-   taste profiling;
-   personalized recommendations;
-   Movie Blend invitation flow;
-   persistent Blend profiles and results;
-   Streamlit deployment.

Potential future work:

-   move more transformations into a formal dbt production layer;
-   scheduled cache maintenance and metadata refreshes;
-   stronger recommendation evaluation;
-   additional observability and pipeline metrics;
-   automated tests and CI;
-   more granular privacy controls for shared Blend data.

------------------------------------------------------------------------

## What This Project Demonstrates

This project was built to practice the complete lifecycle of a data
product rather than only model development or dashboard creation.

It demonstrates:

-   ingestion of user-generated datasets;
-   API integration and external-data enrichment;
-   data cleaning and entity matching;
-   relational database design;
-   PostgreSQL querying and upserts;
-   caching strategies;
-   concurrent I/O;
-   fault-tolerant pipeline behavior;
-   feature engineering and preference modeling;
-   recommendation-system design;
-   application performance optimization;
-   state management;
-   cloud deployment;
-   privacy-aware product decisions.

------------------------------------------------------------------------

## Screenshots

Add final screenshots here before publishing the portfolio version.

Suggested order:

``` text
1. Overview / main analytics
2. Cinema Map
3. Personalized Recommendations
4. Movie Blend comparison
5. Movie Blend shared recommendations
```

Example structure:

``` markdown
### Personal Analytics

![Overview](docs/images/overview.png)

### Recommendations

![Recommendations](docs/images/recommendations.png)

### Movie Blend

![Movie Blend](docs/images/movie-blend.png)
```

------------------------------------------------------------------------

## Author

**Jamille Ghazaleh**

Data Science graduate focused on Data Engineering, BI, analytics, and
data products.

GitHub: https://github.com/jamillerochagh\
LinkedIn: https://www.linkedin.com/in/jamilleghazaleh/

------------------------------------------------------------------------

## Disclaimer

This project is an independent portfolio project and is not affiliated
with Letterboxd or TMDB.

Movie metadata and imagery are enriched using TMDB.
