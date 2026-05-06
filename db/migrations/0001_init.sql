CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS profiles (
    id            BIGSERIAL PRIMARY KEY,
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    public_id     TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    actor_id      TEXT NOT NULL,
    actor_type    TEXT NOT NULL,
    username      CITEXT NOT NULL,
    display_name  TEXT NOT NULL,
    bio           TEXT NOT NULL DEFAULT '',
    avatar_url    TEXT NOT NULL DEFAULT '',
    visibility    TEXT NOT NULL DEFAULT 'public',
    shadow_banned BOOLEAN NOT NULL DEFAULT FALSE,
    banned        BOOLEAN NOT NULL DEFAULT FALSE,
    ban_reason    TEXT NOT NULL DEFAULT '',
    banned_at     TIMESTAMPTZ,
    date_of_birth DATE,
    age_verified  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT profiles_actor_type_chk CHECK (actor_type IN ('person', 'business')),
    CONSTRAINT profiles_visibility_chk CHECK (visibility IN ('public', 'private', 'friends_only')),
    CONSTRAINT profiles_actor_tenant_key UNIQUE (tenant_id, actor_id, actor_type),
    CONSTRAINT profiles_tenant_username_key UNIQUE (tenant_id, username)
);

CREATE TABLE IF NOT EXISTS profile_settings (
    profile_id    BIGINT PRIMARY KEY REFERENCES profiles(id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    allow_follow  BOOLEAN NOT NULL DEFAULT true,
    allow_messages BOOLEAN NOT NULL DEFAULT true,
    discoverable  BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS blocks (
    tenant_id          TEXT NOT NULL DEFAULT 'default',
    blocker_actor_id   TEXT NOT NULL,
    blocker_actor_type TEXT NOT NULL,
    blocked_actor_id   TEXT NOT NULL,
    blocked_actor_type TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, blocker_actor_id, blocker_actor_type, blocked_actor_id, blocked_actor_type),
    CONSTRAINT blocks_blocker_type_chk CHECK (blocker_actor_type IN ('person', 'business')),
    CONSTRAINT blocks_blocked_type_chk CHECK (blocked_actor_type IN ('person', 'business'))
);

CREATE TABLE IF NOT EXISTS follows (
    tenant_id          TEXT NOT NULL DEFAULT 'default',
    follower_actor_id  TEXT NOT NULL,
    follower_actor_type TEXT NOT NULL,
    target_actor_id    TEXT NOT NULL,
    target_actor_type  TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, follower_actor_id, follower_actor_type, target_actor_id, target_actor_type),
    CONSTRAINT follows_follower_type_chk CHECK (follower_actor_type IN ('person', 'business')),
    CONSTRAINT follows_target_type_chk CHECK (target_actor_type IN ('person', 'business'))
);

CREATE TABLE IF NOT EXISTS friendships (
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    actor_a_id   TEXT NOT NULL,
    actor_a_type TEXT NOT NULL,
    actor_b_id   TEXT NOT NULL,
    actor_b_type TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, actor_a_id, actor_a_type, actor_b_id, actor_b_type),
    CONSTRAINT friendships_actor_a_type_chk CHECK (actor_a_type IN ('person', 'business')),
    CONSTRAINT friendships_actor_b_type_chk CHECK (actor_b_type IN ('person', 'business')),
    CONSTRAINT friendships_distinct_chk CHECK (
        actor_a_id <> actor_b_id OR actor_a_type <> actor_b_type
    )
);

CREATE TABLE IF NOT EXISTS friend_requests (
    tenant_id            TEXT NOT NULL DEFAULT 'default',
    requester_actor_id   TEXT NOT NULL,
    requester_actor_type TEXT NOT NULL,
    target_actor_id      TEXT NOT NULL,
    target_actor_type    TEXT NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, requester_actor_id, requester_actor_type, target_actor_id, target_actor_type),
    CONSTRAINT friend_requests_requester_type_chk CHECK (requester_actor_type IN ('person', 'business')),
    CONSTRAINT friend_requests_target_type_chk CHECK (target_actor_type IN ('person', 'business')),
    CONSTRAINT friend_requests_distinct_chk CHECK (
        requester_actor_id <> target_actor_id OR requester_actor_type <> target_actor_type
    )
);

CREATE TABLE IF NOT EXISTS playlists (
    id               BIGSERIAL PRIMARY KEY,
    tenant_id        TEXT NOT NULL DEFAULT 'default',
    public_id        TEXT NOT NULL,
    owner_user_id    TEXT NOT NULL,
    owner_actor_id   TEXT NOT NULL,
    owner_actor_type TEXT NOT NULL,
    title            TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    cover_url        TEXT NOT NULL DEFAULT '',
    visibility       TEXT NOT NULL DEFAULT 'public',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT playlists_owner_actor_type_chk CHECK (owner_actor_type IN ('person', 'business')),
    CONSTRAINT playlists_visibility_chk CHECK (visibility IN ('public', 'private', 'friends_only', 'shared', 'link'))
);

CREATE TABLE IF NOT EXISTS playlist_items (
    id                  BIGSERIAL PRIMARY KEY,
    tenant_id           TEXT NOT NULL DEFAULT 'default',
    public_id           TEXT NOT NULL,
    playlist_id         BIGINT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    position            INT NOT NULL,
    place_id            TEXT NOT NULL DEFAULT '',
    external_source     TEXT NOT NULL DEFAULT '',
    external_id         TEXT NOT NULL DEFAULT '',
    name_snapshot       TEXT NOT NULL,
    address_snapshot    TEXT NOT NULL DEFAULT '',
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    note                TEXT NOT NULL DEFAULT '',
    added_by_actor_id   TEXT NOT NULL,
    added_by_actor_type TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT playlist_items_added_by_actor_type_chk CHECK (added_by_actor_type IN ('person', 'business')),
    CONSTRAINT playlist_items_position_positive_chk CHECK (position > 0),
    CONSTRAINT playlist_items_name_snapshot_nonempty_chk CHECK (length(trim(name_snapshot)) > 0)
);

CREATE TABLE IF NOT EXISTS playlist_shares (
    tenant_id        TEXT NOT NULL DEFAULT 'default',
    playlist_id      BIGINT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    shared_actor_id  TEXT NOT NULL,
    shared_actor_type TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, playlist_id, shared_actor_id, shared_actor_type),
    CONSTRAINT playlist_shares_actor_type_chk CHECK (shared_actor_type IN ('person', 'business'))
);

CREATE TABLE IF NOT EXISTS playlist_collaborators (
    tenant_id  TEXT NOT NULL DEFAULT 'default',
    playlist_id BIGINT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    actor_id   TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    role       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, playlist_id, actor_id, actor_type),
    CONSTRAINT chk_playlist_collaborators_actor_type CHECK (actor_type IN ('person', 'business')),
    CONSTRAINT chk_playlist_collaborators_role CHECK (role IN ('editor'))
);

CREATE TABLE IF NOT EXISTS events (
    id               BIGSERIAL PRIMARY KEY,
    tenant_id        TEXT NOT NULL DEFAULT 'default',
    public_id        TEXT NOT NULL,
    owner_user_id    TEXT NOT NULL,
    owner_actor_id   TEXT NOT NULL,
    owner_actor_type TEXT NOT NULL,
    title            TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    visibility       TEXT NOT NULL,
    status           TEXT NOT NULL,
    start_at         TIMESTAMPTZ NOT NULL,
    end_at           TIMESTAMPTZ,
    timezone         TEXT NOT NULL,
    place_id         TEXT NOT NULL DEFAULT '',
    external_source  TEXT NOT NULL DEFAULT '',
    external_id      TEXT NOT NULL DEFAULT '',
    name_snapshot    TEXT NOT NULL,
    address_snapshot TEXT NOT NULL DEFAULT '',
    latitude         DOUBLE PRECISION,
    longitude        DOUBLE PRECISION,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_events_actor_type CHECK (owner_actor_type IN ('person', 'business')),
    CONSTRAINT chk_events_visibility CHECK (visibility IN ('public', 'private', 'friends_only', 'invite_only', 'participants_only')),
    CONSTRAINT chk_events_status CHECK (status IN ('draft', 'voting', 'confirmed', 'finalized', 'cancelled')),
    CONSTRAINT chk_events_time_range CHECK (end_at IS NULL OR end_at >= start_at)
);

CREATE TABLE IF NOT EXISTS event_participants (
    tenant_id         TEXT NOT NULL DEFAULT 'default',
    event_id          BIGINT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    actor_id          TEXT NOT NULL,
    actor_type        TEXT NOT NULL,
    role              TEXT NOT NULL,
    attendance_status TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, event_id, actor_id, actor_type),
    CONSTRAINT chk_event_participants_actor_type CHECK (actor_type IN ('person', 'business')),
    CONSTRAINT chk_event_participants_role CHECK (role IN ('owner', 'organizer', 'participant')),
    CONSTRAINT chk_event_participants_attendance CHECK (attendance_status IN ('invited', 'going', 'maybe', 'declined'))
);

CREATE TABLE IF NOT EXISTS profile_place_entries (
    id               BIGSERIAL PRIMARY KEY,
    tenant_id        TEXT NOT NULL DEFAULT 'default',
    public_id        TEXT NOT NULL,
    owner_user_id    TEXT NOT NULL,
    owner_actor_id   TEXT NOT NULL,
    owner_actor_type TEXT NOT NULL,
    entry_type       TEXT NOT NULL,
    place_ref        TEXT NOT NULL,
    place_id         TEXT NOT NULL DEFAULT '',
    external_source  TEXT NOT NULL DEFAULT '',
    external_id      TEXT NOT NULL DEFAULT '',
    name_snapshot    TEXT NOT NULL,
    address_snapshot TEXT NOT NULL DEFAULT '',
    latitude         DOUBLE PRECISION,
    longitude        DOUBLE PRECISION,
    note             TEXT NOT NULL DEFAULT '',
    visibility       TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_profile_place_entries_actor_type CHECK (owner_actor_type IN ('person', 'business')),
    CONSTRAINT chk_profile_place_entries_type CHECK (entry_type IN ('favorite', 'want_to_go', 'been_there')),
    CONSTRAINT chk_profile_place_entries_visibility CHECK (visibility IN ('public', 'private', 'friends_only')),
    CONSTRAINT uq_profile_place_entries_owner_type_place UNIQUE (owner_actor_id, owner_actor_type, entry_type, place_ref)
);

CREATE TABLE IF NOT EXISTS profile_reviews (
    id               BIGSERIAL PRIMARY KEY,
    tenant_id        TEXT NOT NULL DEFAULT 'default',
    public_id        TEXT NOT NULL,
    owner_user_id    TEXT NOT NULL,
    owner_actor_id   TEXT NOT NULL,
    owner_actor_type TEXT NOT NULL,
    place_ref        TEXT NOT NULL,
    place_id         TEXT NOT NULL DEFAULT '',
    external_source  TEXT NOT NULL DEFAULT '',
    external_id      TEXT NOT NULL DEFAULT '',
    name_snapshot    TEXT NOT NULL,
    address_snapshot TEXT NOT NULL DEFAULT '',
    rating           INTEGER NOT NULL,
    title            TEXT NOT NULL DEFAULT '',
    body             TEXT NOT NULL DEFAULT '',
    visibility       TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_profile_reviews_actor_type CHECK (owner_actor_type IN ('person', 'business')),
    CONSTRAINT chk_profile_reviews_visibility CHECK (visibility IN ('public', 'private', 'friends_only')),
    CONSTRAINT chk_profile_reviews_rating CHECK (rating BETWEEN 1 AND 5),
    CONSTRAINT uq_profile_reviews_owner_place UNIQUE (owner_actor_id, owner_actor_type, place_ref)
);

CREATE TABLE IF NOT EXISTS posts (
    id               BIGSERIAL PRIMARY KEY,
    tenant_id        TEXT NOT NULL DEFAULT 'default',
    public_id        TEXT NOT NULL,
    owner_user_id    TEXT NOT NULL,
    owner_actor_id   TEXT NOT NULL,
    owner_actor_type TEXT NOT NULL,
    caption          TEXT NOT NULL DEFAULT '',
    visibility       TEXT NOT NULL,
    place_id         TEXT NOT NULL DEFAULT '',
    external_source  TEXT NOT NULL DEFAULT '',
    external_id      TEXT NOT NULL DEFAULT '',
    name_snapshot    TEXT NOT NULL DEFAULT '',
    address_snapshot TEXT NOT NULL DEFAULT '',
    latitude         DOUBLE PRECISION,
    longitude        DOUBLE PRECISION,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_posts_actor_type CHECK (owner_actor_type IN ('person', 'business')),
    CONSTRAINT chk_posts_visibility CHECK (visibility IN ('public', 'private', 'friends_only', 'followers_only'))
);

CREATE TABLE IF NOT EXISTS post_media (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    public_id       TEXT NOT NULL,
    post_id         BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    position        INTEGER NOT NULL,
    media_type      TEXT NOT NULL,
    media_url       TEXT NOT NULL,
    thumbnail_url   TEXT NOT NULL DEFAULT '',
    duration_seconds INTEGER,
    asset_public_id TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_post_media_type CHECK (media_type IN ('image', 'video')),
    CONSTRAINT chk_post_media_position CHECK (position >= 0),
    CONSTRAINT chk_post_media_duration CHECK (duration_seconds IS NULL OR duration_seconds > 0),
    CONSTRAINT uq_post_media_position UNIQUE (post_id, position)
);

CREATE TABLE IF NOT EXISTS post_likes (
    tenant_id  TEXT NOT NULL DEFAULT 'default',
    post_id    BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    actor_id   TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, post_id, actor_id, actor_type),
    CONSTRAINT chk_post_likes_actor_type CHECK (actor_type IN ('person', 'business'))
);

CREATE TABLE IF NOT EXISTS post_saves (
    tenant_id  TEXT NOT NULL DEFAULT 'default',
    post_id    BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    actor_id   TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, post_id, actor_id, actor_type),
    CONSTRAINT chk_post_saves_actor_type CHECK (actor_type IN ('person', 'business'))
);

CREATE TABLE IF NOT EXISTS post_views (
    tenant_id      TEXT NOT NULL DEFAULT 'default',
    post_id        BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    actor_id       TEXT NOT NULL,
    actor_type     TEXT NOT NULL,
    view_count     BIGINT NOT NULL DEFAULT 1,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_viewed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, post_id, actor_id, actor_type),
    CONSTRAINT chk_post_views_actor_type CHECK (actor_type IN ('person', 'business')),
    CONSTRAINT chk_post_views_count CHECK (view_count > 0)
);

CREATE TABLE IF NOT EXISTS post_comments (
    id                BIGSERIAL PRIMARY KEY,
    tenant_id         TEXT NOT NULL DEFAULT 'default',
    public_id         TEXT NOT NULL UNIQUE,
    post_id           BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    parent_comment_id BIGINT REFERENCES post_comments(id) ON DELETE CASCADE,
    owner_user_id     TEXT NOT NULL,
    owner_actor_id    TEXT NOT NULL,
    owner_actor_type  TEXT NOT NULL,
    body              TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_post_comments_actor_type CHECK (owner_actor_type IN ('person', 'business'))
);

CREATE TABLE IF NOT EXISTS post_shares (
    tenant_id  TEXT NOT NULL DEFAULT 'default',
    public_id  TEXT NOT NULL,
    post_id    BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    actor_id   TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, post_id, actor_id, actor_type),
    CONSTRAINT chk_post_shares_actor_type CHECK (actor_type IN ('person', 'business'))
);

CREATE TABLE IF NOT EXISTS media_assets (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    public_id       TEXT NOT NULL UNIQUE,
    owner_user_id   TEXT NOT NULL,
    owner_actor_id  TEXT NOT NULL,
    owner_actor_type TEXT NOT NULL,
    media_type      TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    media_url       TEXT NOT NULL DEFAULT '',
    thumbnail_url   TEXT NOT NULL DEFAULT '',
    duration_seconds INTEGER,
    status          TEXT NOT NULL DEFAULT 'pending',
    failure_reason  TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at    TIMESTAMPTZ,
    CONSTRAINT chk_media_assets_actor_type CHECK (owner_actor_type IN ('person', 'business')),
    CONSTRAINT chk_media_assets_type CHECK (media_type IN ('image', 'video')),
    CONSTRAINT chk_media_assets_status CHECK (status IN ('pending', 'processing', 'ready', 'failed'))
);

CREATE TABLE IF NOT EXISTS post_moderation (
    post_id     BIGINT PRIMARY KEY REFERENCES posts(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    status      TEXT NOT NULL DEFAULT 'active',
    reason_code TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at TIMESTAMPTZ,
    CONSTRAINT chk_post_moderation_status CHECK (status IN ('active', 'hidden', 'removed'))
);

CREATE TABLE IF NOT EXISTS post_comment_moderation (
    comment_id  BIGINT PRIMARY KEY REFERENCES post_comments(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    status      TEXT NOT NULL DEFAULT 'active',
    reason_code TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at TIMESTAMPTZ,
    CONSTRAINT chk_post_comment_moderation_status CHECK (status IN ('active', 'hidden', 'removed'))
);

CREATE TABLE IF NOT EXISTS content_reports (
    id                  BIGSERIAL PRIMARY KEY,
    tenant_id           TEXT NOT NULL DEFAULT 'default',
    public_id           TEXT NOT NULL UNIQUE,
    target_type         TEXT NOT NULL,
    target_public_id    TEXT NOT NULL,
    reporter_user_id    TEXT NOT NULL,
    reporter_actor_id   TEXT NOT NULL,
    reporter_actor_type TEXT NOT NULL,
    reason_code         TEXT NOT NULL,
    detail              TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'open',
    resolution_note     TEXT NOT NULL DEFAULT '',
    reviewed_by         TEXT NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at         TIMESTAMPTZ,
    CONSTRAINT chk_content_reports_target CHECK (target_type IN ('post', 'comment', 'place', 'profile')),
    CONSTRAINT chk_content_reports_actor_type CHECK (reporter_actor_type IN ('person', 'business')),
    CONSTRAINT chk_content_reports_status CHECK (status IN ('open', 'reviewed', 'dismissed', 'actioned'))
);

CREATE TABLE IF NOT EXISTS places (
    id                   BIGSERIAL PRIMARY KEY,
    tenant_id            TEXT NOT NULL DEFAULT 'default',
    public_id            TEXT NOT NULL UNIQUE,
    owner_user_id        TEXT NOT NULL,
    owner_actor_id       TEXT NOT NULL,
    owner_actor_type     TEXT NOT NULL,
    name                 TEXT NOT NULL,
    category             TEXT NOT NULL,
    description          TEXT NOT NULL DEFAULT '',
    about                TEXT NOT NULL DEFAULT '',
    address_line         TEXT NOT NULL DEFAULT '',
    city                 TEXT NOT NULL DEFAULT '',
    region               TEXT NOT NULL DEFAULT '',
    country_code         TEXT NOT NULL DEFAULT '',
    postal_code          TEXT NOT NULL DEFAULT '',
    latitude             DOUBLE PRECISION,
    longitude            DOUBLE PRECISION,
    rating_average       DOUBLE PRECISION NOT NULL DEFAULT 0,
    ratings_count        BIGINT NOT NULL DEFAULT 0,
    price_level          INTEGER NOT NULL DEFAULT 0,
    external_source      TEXT NOT NULL DEFAULT '',
    external_id          TEXT NOT NULL DEFAULT '',
    website_url          TEXT NOT NULL DEFAULT '',
    phone                TEXT NOT NULL DEFAULT '',
    verified             BOOLEAN NOT NULL DEFAULT FALSE,
    status               TEXT NOT NULL DEFAULT 'active',
    is_accessible        BOOLEAN NOT NULL DEFAULT FALSE,
    is_outdoor           BOOLEAN NOT NULL DEFAULT FALSE,
    is_family_friendly   BOOLEAN NOT NULL DEFAULT FALSE,
    is_pet_friendly      BOOLEAN NOT NULL DEFAULT FALSE,
    has_parking          BOOLEAN NOT NULL DEFAULT FALSE,
    has_wifi             BOOLEAN NOT NULL DEFAULT FALSE,
    serves_alcohol       BOOLEAN NOT NULL DEFAULT FALSE,
    accepts_reservations BOOLEAN NOT NULL DEFAULT FALSE,
    adult_content        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT places_status_check CHECK (status IN ('active', 'hidden')),
    CONSTRAINT places_price_level_check CHECK (price_level >= 0 AND price_level <= 4)
);

CREATE TABLE IF NOT EXISTS place_media (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    public_id       TEXT NOT NULL UNIQUE,
    place_id        BIGINT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    asset_public_id TEXT NOT NULL REFERENCES media_assets(public_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL,
    media_type      TEXT NOT NULL,
    media_url       TEXT NOT NULL,
    thumbnail_url   TEXT NOT NULL DEFAULT '',
    duration_seconds INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT place_media_position_check CHECK (position >= 0)
);

CREATE TABLE IF NOT EXISTS place_highlights (
    id         BIGSERIAL PRIMARY KEY,
    tenant_id  TEXT NOT NULL,
    place_id   BIGINT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    slug       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_place_highlights_place_slug UNIQUE (place_id, slug)
);

CREATE TABLE IF NOT EXISTS place_recommended_for (
    id         BIGSERIAL PRIMARY KEY,
    tenant_id  TEXT NOT NULL,
    place_id   BIGINT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    slug       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_place_recommended_for_place_slug UNIQUE (place_id, slug)
);

CREATE TABLE IF NOT EXISTS place_hours (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    place_id    BIGINT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    day_of_week SMALLINT NOT NULL,
    opens_at    TEXT NOT NULL DEFAULT '',
    closes_at   TEXT NOT NULL DEFAULT '',
    is_closed   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT place_hours_day_of_week_check CHECK (day_of_week >= 0 AND day_of_week <= 6),
    CONSTRAINT uq_place_hours_place_day UNIQUE (place_id, day_of_week)
);

CREATE TABLE IF NOT EXISTS place_likes (
    tenant_id  TEXT NOT NULL,
    place_id   BIGINT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    actor_id   TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, place_id, actor_id, actor_type)
);

CREATE TABLE IF NOT EXISTS place_dislikes (
    tenant_id  TEXT NOT NULL,
    place_id   BIGINT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    actor_id   TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, place_id, actor_id, actor_type)
);

CREATE TABLE IF NOT EXISTS telemetry_events (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    user_id     TEXT NOT NULL,
    actor_id    TEXT NOT NULL,
    actor_type  TEXT NOT NULL DEFAULT 'person',
    event_type  TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT '',
    entity_id   TEXT NOT NULL DEFAULT '',
    payload     JSONB NOT NULL DEFAULT '{}',
    session_id  TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_telemetry_event_type CHECK (
        event_type IN (
            'impression', 'view_time', 'scroll_depth',
            'search_query', 'feed_position_click',
            'place_detail_view', 'category_browse',
            'filter_applied', 'share_intent',
            'route_click', 'content_view', 'content_like',
            'content_create', 'visit_feedback'
        )
    ),
    CONSTRAINT chk_telemetry_entity_type CHECK (
        entity_type IN ('', 'place', 'post', 'event', 'playlist', 'profile', 'content')
    )
);

CREATE TABLE IF NOT EXISTS onboarding_responses (
    id            BIGSERIAL PRIMARY KEY,
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    user_id       TEXT NOT NULL,
    actor_id      TEXT NOT NULL,
    actor_type    TEXT NOT NULL DEFAULT 'person',
    question_key  TEXT NOT NULL,
    answer_values TEXT[] NOT NULL DEFAULT '{}',
    completed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_onboarding_response UNIQUE (tenant_id, user_id, question_key),
    CONSTRAINT chk_onboarding_question_key CHECK (
        question_key IN (
            'preferred_categories', 'preferred_price_levels',
            'preferred_vibes', 'preferred_for',
            'preferred_style', 'preferred_distance'
        )
    )
);

CREATE TABLE IF NOT EXISTS checkins (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    public_id       TEXT NOT NULL UNIQUE,
    user_id         TEXT NOT NULL,
    actor_id        TEXT NOT NULL,
    actor_type      TEXT NOT NULL DEFAULT 'person',
    place_id        BIGINT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    distance_meters DOUBLE PRECISION NOT NULL,
    verified        BOOLEAN NOT NULL DEFAULT FALSE,
    note            TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_preference_profiles (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    user_id         TEXT NOT NULL,
    actor_id        TEXT NOT NULL,
    actor_type      TEXT NOT NULL DEFAULT 'person',
    profile_type    TEXT NOT NULL,
    dimension       TEXT NOT NULL,
    dimension_value TEXT NOT NULL,
    weight          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_user_pref UNIQUE (tenant_id, user_id, profile_type, dimension, dimension_value),
    CONSTRAINT chk_profile_type CHECK (profile_type IN ('base', 'recent', 'composition')),
    CONSTRAINT chk_dimension CHECK (dimension IN ('category', 'price_level', 'highlight', 'recommended_for', 'weight'))
);

CREATE TABLE IF NOT EXISTS place_sync_jobs (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    external_source TEXT NOT NULL,
    query           TEXT NOT NULL DEFAULT '',
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    radius_meters   DOUBLE PRECISION,
    status          TEXT NOT NULL DEFAULT 'pending',
    results_count   INTEGER NOT NULL DEFAULT 0,
    created_count   INTEGER NOT NULL DEFAULT 0,
    updated_count   INTEGER NOT NULL DEFAULT 0,
    skipped_count   INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT NOT NULL DEFAULT '',
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_sync_status CHECK (status IN ('pending', 'running', 'completed', 'failed'))
);

CREATE TABLE IF NOT EXISTS place_sync_log (
    id              BIGSERIAL PRIMARY KEY,
    job_id          BIGINT NOT NULL REFERENCES place_sync_jobs(id) ON DELETE CASCADE,
    external_source TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    action          TEXT NOT NULL,
    place_id        BIGINT,
    detail          TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_sync_log_action CHECK (action IN ('created', 'updated', 'skipped', 'error'))
);

CREATE TABLE IF NOT EXISTS analytics_snapshots (
    id           BIGSERIAL PRIMARY KEY,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    metric_key   TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    dimensions   JSONB NOT NULL DEFAULT '{}',
    period_start TIMESTAMPTZ NOT NULL,
    period_end   TIMESTAMPTZ NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_analytics_snapshot UNIQUE (tenant_id, metric_key, period_start, dimensions)
);

CREATE TABLE IF NOT EXISTS user_bandit_state (
    tenant_id         TEXT NOT NULL DEFAULT 'default',
    user_id           TEXT NOT NULL,
    base_alpha        DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    base_beta         DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    recent_alpha      DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    recent_beta       DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    exploratory_alpha DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    exploratory_beta  DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id)
);

CREATE TABLE IF NOT EXISTS ai_run_metadata (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    run_type        TEXT NOT NULL,
    last_signal_at  TIMESTAMPTZ,
    users_processed INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_run_type CHECK (run_type IN ('full_recompute', 'incremental', 'train', 'bandit_update'))
);

CREATE TABLE IF NOT EXISTS user_session_context (
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    user_id         TEXT NOT NULL,
    companion       TEXT NOT NULL DEFAULT '',
    budget          TEXT NOT NULL DEFAULT '',
    time_of_day     TEXT NOT NULL DEFAULT '',
    intent          TEXT NOT NULL DEFAULT '',
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    max_distance_km DOUBLE PRECISION,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id)
);

CREATE TABLE IF NOT EXISTS ai_write_audit (
    id            BIGSERIAL PRIMARY KEY,
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    user_id       TEXT NOT NULL,
    operation     TEXT NOT NULL,
    profile_type  TEXT NOT NULL DEFAULT '',
    rows_affected INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_profiles_actor
    ON profiles(tenant_id, actor_id, actor_type);
CREATE UNIQUE INDEX IF NOT EXISTS uq_profiles_public_id
    ON profiles(public_id);
CREATE INDEX IF NOT EXISTS idx_profiles_shadow_banned
    ON profiles(tenant_id) WHERE shadow_banned = TRUE;
CREATE INDEX IF NOT EXISTS idx_profiles_banned
    ON profiles(tenant_id) WHERE banned = TRUE;

CREATE INDEX IF NOT EXISTS idx_blocks_blocked_tenant
    ON blocks(tenant_id, blocked_actor_id, blocked_actor_type);

CREATE INDEX IF NOT EXISTS idx_follows_target_tenant
    ON follows(tenant_id, target_actor_id, target_actor_type);

CREATE INDEX IF NOT EXISTS idx_friendships_actor_b_tenant
    ON friendships(tenant_id, actor_b_id, actor_b_type);

CREATE INDEX IF NOT EXISTS idx_friend_requests_target_tenant
    ON friend_requests(tenant_id, target_actor_id, target_actor_type);
CREATE UNIQUE INDEX IF NOT EXISTS uq_friend_requests_tenant_actor_pair
    ON friend_requests (
        tenant_id,
        LEAST(requester_actor_type || ':' || requester_actor_id, target_actor_type || ':' || target_actor_id),
        GREATEST(requester_actor_type || ':' || requester_actor_id, target_actor_type || ':' || target_actor_id)
    );

CREATE INDEX IF NOT EXISTS idx_playlists_owner_tenant
    ON playlists(tenant_id, owner_actor_id, owner_actor_type);
CREATE UNIQUE INDEX IF NOT EXISTS uq_playlists_public_id
    ON playlists(public_id);
CREATE INDEX IF NOT EXISTS idx_playlists_tenant_public_id
    ON playlists(tenant_id, public_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_playlist_items_position
    ON playlist_items(playlist_id, position);
CREATE INDEX IF NOT EXISTS idx_playlist_items_playlist_tenant
    ON playlist_items(tenant_id, playlist_id, position);
CREATE UNIQUE INDEX IF NOT EXISTS uq_playlist_items_public_id
    ON playlist_items(public_id);

CREATE INDEX IF NOT EXISTS idx_playlist_shares_actor_tenant
    ON playlist_shares(tenant_id, shared_actor_id, shared_actor_type);

CREATE INDEX IF NOT EXISTS idx_playlist_collaborators_actor_tenant
    ON playlist_collaborators(tenant_id, actor_id, actor_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_owner_tenant
    ON events(tenant_id, owner_actor_id, owner_actor_type, start_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_events_public_id
    ON events(public_id);
CREATE INDEX IF NOT EXISTS idx_events_tenant_public_id
    ON events(tenant_id, public_id);

CREATE INDEX IF NOT EXISTS idx_event_participants_actor_tenant
    ON event_participants(tenant_id, actor_id, actor_type, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_profile_place_entries_owner_tenant
    ON profile_place_entries(tenant_id, owner_actor_id, owner_actor_type, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_profile_place_entries_public_id
    ON profile_place_entries(public_id);
CREATE INDEX IF NOT EXISTS idx_profile_place_entries_tenant_public_id
    ON profile_place_entries(tenant_id, public_id);

CREATE INDEX IF NOT EXISTS idx_profile_reviews_owner_tenant
    ON profile_reviews(tenant_id, owner_actor_id, owner_actor_type, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_profile_reviews_public_id
    ON profile_reviews(public_id);
CREATE INDEX IF NOT EXISTS idx_profile_reviews_tenant_public_id
    ON profile_reviews(tenant_id, public_id);

CREATE INDEX IF NOT EXISTS idx_posts_owner_tenant
    ON posts(tenant_id, owner_actor_id, owner_actor_type, id DESC);
CREATE INDEX IF NOT EXISTS idx_posts_feed_tenant
    ON posts(tenant_id, id DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_posts_public_id
    ON posts(public_id);
CREATE INDEX IF NOT EXISTS idx_posts_tenant_public_id
    ON posts(tenant_id, public_id);

CREATE INDEX IF NOT EXISTS idx_post_media_post_tenant
    ON post_media(tenant_id, post_id, position ASC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_post_media_public_id
    ON post_media(public_id);

CREATE INDEX IF NOT EXISTS idx_post_likes_actor_tenant
    ON post_likes(tenant_id, actor_id, actor_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_post_saves_actor_tenant
    ON post_saves(tenant_id, actor_id, actor_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_post_views_actor_tenant
    ON post_views(tenant_id, actor_id, actor_type, last_viewed_at DESC);

CREATE INDEX IF NOT EXISTS idx_post_comments_post_tenant
    ON post_comments(tenant_id, post_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_post_comments_owner_tenant
    ON post_comments(tenant_id, owner_actor_id, owner_actor_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_post_comments_post_parent
    ON post_comments(post_id, parent_comment_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_post_comments_tenant_public_id
    ON post_comments(tenant_id, public_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_post_shares_public_id
    ON post_shares(public_id);
CREATE INDEX IF NOT EXISTS idx_post_shares_created
    ON post_shares(created_at DESC, public_id DESC);
CREATE INDEX IF NOT EXISTS idx_post_shares_actor_tenant
    ON post_shares(tenant_id, actor_id, actor_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_post_shares_tenant_public_id
    ON post_shares(tenant_id, public_id);

CREATE INDEX IF NOT EXISTS idx_media_assets_owner_tenant
    ON media_assets(tenant_id, owner_actor_id, owner_actor_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_media_assets_tenant_public_id
    ON media_assets(tenant_id, public_id);

CREATE INDEX IF NOT EXISTS idx_content_reports_target_tenant
    ON content_reports(tenant_id, target_type, target_public_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_content_reports_status
    ON content_reports(status, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_content_reports_open_per_reporter
    ON content_reports(tenant_id, reporter_user_id, target_type, target_public_id)
    WHERE status = 'open';

CREATE INDEX IF NOT EXISTS idx_places_owner_tenant
    ON places(tenant_id, owner_actor_id, owner_actor_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_places_status_tenant
    ON places(tenant_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_places_category_tenant
    ON places(tenant_id, category, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_places_coordinates_tenant
    ON places(tenant_id, latitude, longitude)
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_places_external_ref_tenant
    ON places(tenant_id, external_source, external_id)
    WHERE external_source <> '' AND external_id <> '';
CREATE INDEX IF NOT EXISTS idx_places_tenant_public_id
    ON places(tenant_id, public_id);
CREATE INDEX IF NOT EXISTS idx_places_tenant_created
    ON places(tenant_id, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_places_adult_content
    ON places(tenant_id, adult_content) WHERE adult_content = true;

CREATE UNIQUE INDEX IF NOT EXISTS uq_place_media_asset_per_place
    ON place_media(place_id, asset_public_id);
CREATE INDEX IF NOT EXISTS idx_place_media_place_tenant
    ON place_media(tenant_id, place_id, position ASC);
CREATE INDEX IF NOT EXISTS idx_place_media_asset_public_id
    ON place_media(asset_public_id);

CREATE INDEX IF NOT EXISTS idx_place_highlights_tenant_slug
    ON place_highlights(tenant_id, slug);

CREATE INDEX IF NOT EXISTS idx_place_recommended_for_tenant_slug
    ON place_recommended_for(tenant_id, slug);

CREATE INDEX IF NOT EXISTS idx_place_hours_tenant_day
    ON place_hours(tenant_id, day_of_week);

CREATE INDEX IF NOT EXISTS idx_place_likes_place_tenant
    ON place_likes(place_id, tenant_id);

CREATE INDEX IF NOT EXISTS idx_place_dislikes_place_tenant
    ON place_dislikes(place_id, tenant_id);

CREATE INDEX IF NOT EXISTS idx_telemetry_user_time
    ON telemetry_events(tenant_id, user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_entity
    ON telemetry_events(tenant_id, entity_type, entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_type_time
    ON telemetry_events(tenant_id, event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_onboarding_user
    ON onboarding_responses(tenant_id, user_id);

CREATE INDEX IF NOT EXISTS idx_checkins_user_time
    ON checkins(tenant_id, user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_checkins_place_time
    ON checkins(tenant_id, place_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_checkins_user_place
    ON checkins(tenant_id, user_id, place_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_checkins_user_place_day
    ON checkins(tenant_id, user_id, place_id, (timezone('UTC', created_at)::date));

CREATE INDEX IF NOT EXISTS idx_user_pref_user
    ON user_preference_profiles(tenant_id, user_id, profile_type);

CREATE INDEX IF NOT EXISTS idx_place_sync_jobs_tenant
    ON place_sync_jobs(tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analytics_metric
    ON analytics_snapshots(tenant_id, metric_key, period_start DESC);

CREATE INDEX IF NOT EXISTS idx_ai_run_metadata_tenant
    ON ai_run_metadata(tenant_id, run_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ai_write_audit_tenant
    ON ai_write_audit(tenant_id, created_at DESC);

-- ============================================================
-- ROW LEVEL SECURITY
-- ============================================================

DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOR tbl IN
        SELECT unnest(ARRAY[
            'profiles',
            'profile_settings',
            'blocks',
            'follows',
            'friendships',
            'friend_requests',
            'playlists',
            'playlist_items',
            'playlist_shares',
            'playlist_collaborators',
            'events',
            'event_participants',
            'profile_place_entries',
            'profile_reviews',
            'posts',
            'post_media',
            'post_likes',
            'post_saves',
            'post_views',
            'post_comments',
            'post_shares',
            'media_assets',
            'content_reports',
            'places',
            'place_media',
            'place_highlights',
            'place_recommended_for',
            'place_hours',
            'place_likes',
            'place_dislikes',
            'post_moderation',
            'post_comment_moderation',
            'telemetry_events',
            'onboarding_responses',
            'checkins',
            'user_preference_profiles',
            'analytics_snapshots',
            'user_bandit_state',
            'ai_run_metadata',
            'ai_write_audit',
            'user_session_context',
            'place_sync_jobs'
        ])
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tbl);
        EXECUTE format(
            'CREATE POLICY tenant_isolation_%I ON %I
                USING (tenant_id = current_setting(''app.current_tenant'', true))
                WITH CHECK (tenant_id = current_setting(''app.current_tenant'', true))',
            tbl, tbl
        );
    END LOOP;
END
$$;

ALTER TABLE place_sync_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_place_sync_log ON place_sync_log
    USING (job_id IN (SELECT id FROM place_sync_jobs WHERE tenant_id = current_setting('app.current_tenant', true)))
    WITH CHECK (job_id IN (SELECT id FROM place_sync_jobs WHERE tenant_id = current_setting('app.current_tenant', true)));

-- ============================================================
-- GRANTS: ai_worker
-- ============================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_worker') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO ai_worker';

        EXECUTE 'GRANT SELECT ON
            telemetry_events, places, place_highlights, place_recommended_for,
            place_hours, place_media, profile_place_entries, profile_reviews,
            checkins, posts, post_media, post_likes, post_saves, post_views,
            post_comments, post_shares, place_likes, place_dislikes,
            onboarding_responses, profiles, profile_settings,
            events, event_participants, playlists, playlist_items,
            blocks, follows, friendships, friend_requests,
            media_assets
        TO ai_worker';

        EXECUTE 'GRANT SELECT, INSERT, DELETE ON user_preference_profiles, analytics_snapshots TO ai_worker';
        EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON user_bandit_state TO ai_worker';

        EXECUTE 'GRANT SELECT, INSERT ON ai_run_metadata TO ai_worker';

        EXECUTE 'GRANT INSERT ON ai_write_audit TO ai_worker';

        EXECUTE 'GRANT SELECT ON user_session_context TO ai_worker';

        EXECUTE 'GRANT USAGE, SELECT ON
            user_preference_profiles_id_seq, analytics_snapshots_id_seq,
            ai_run_metadata_id_seq, ai_write_audit_id_seq
        TO ai_worker';
    END IF;
END
$$;

-- ============================================================
-- GRANTS: ai_seed (restricted INSERT/DELETE/SELECT for seeding)
-- ============================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_seed') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO ai_seed';

        EXECUTE 'GRANT SELECT, INSERT, DELETE ON
            places, place_highlights, place_recommended_for, place_hours,
            place_media, telemetry_events, place_likes, place_dislikes,
            profile_place_entries, onboarding_responses,
            user_preference_profiles, analytics_snapshots
        TO ai_seed';

        EXECUTE 'GRANT SELECT, INSERT ON
            profiles, profile_reviews, checkins
        TO ai_seed';

        EXECUTE 'GRANT USAGE, SELECT ON
            places_id_seq, place_highlights_id_seq, place_recommended_for_id_seq,
            place_hours_id_seq, place_media_id_seq, telemetry_events_id_seq,
            profile_place_entries_id_seq, onboarding_responses_id_seq,
            user_preference_profiles_id_seq, analytics_snapshots_id_seq,
            profiles_id_seq, profile_reviews_id_seq, checkins_id_seq
        TO ai_seed';

        RAISE NOTICE 'ai_seed role grants applied successfully';
    END IF;
END
$$;

-- ============================================================
-- GRANTS: social_app
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'social_app') THEN
        RAISE NOTICE 'social_app role does not exist yet — skipping grants (will be created by postgres-init.sh)';
        RETURN;
    END IF;

    EXECUTE 'GRANT USAGE ON SCHEMA public TO social_app';

    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO social_app';

    EXECUTE 'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO social_app';

    EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO social_app';
    EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO social_app';

    RAISE NOTICE 'social_app role grants applied successfully';
END
$$;
