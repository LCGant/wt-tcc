-- Force RLS on all tenant-scoped tables so that even the table owner role
-- is subject to row-level security policies. Without FORCE, PostgreSQL
-- silently bypasses RLS for the table owner, which defeats tenant isolation
-- if the application connects with the owner role.

DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOR tbl IN
        SELECT unnest(ARRAY[
            'profiles', 'profile_settings', 'blocks', 'follows',
            'friendships', 'friend_requests', 'playlists', 'playlist_items',
            'playlist_shares', 'playlist_collaborators', 'events',
            'event_participants', 'profile_place_entries', 'profile_reviews',
            'posts', 'post_media', 'post_likes', 'post_saves', 'post_views',
            'post_comments', 'post_shares', 'media_assets', 'content_reports',
            'places', 'place_media', 'place_highlights', 'place_recommended_for',
            'place_hours', 'place_likes', 'place_dislikes',
            'post_moderation', 'post_comment_moderation',
            'telemetry_events', 'onboarding_responses', 'checkins',
            'user_preference_profiles', 'analytics_snapshots',
            'user_bandit_state', 'ai_run_metadata', 'ai_write_audit',
            'user_session_context', 'place_sync_jobs', 'place_sync_log'
        ])
    LOOP
        EXECUTE format('ALTER TABLE IF EXISTS %I FORCE ROW LEVEL SECURITY', tbl);
    END LOOP;
END
$$;
