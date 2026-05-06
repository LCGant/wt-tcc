"""Generate synthetic user interactions based on personas."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone


def generate_interactions(
    users: list[dict],
    places: list[dict],
    interactions_per_user: int,
    days: int,
    rng: random.Random,
) -> dict:
    """
    Generate interactions for each user based on their persona.

    Returns dict of DataFrames:
      onboarding, library_entries, reviews, place_likes, place_dislikes,
      checkins, telemetry_events
    """
    now = datetime.now(timezone.utc)
    place_by_cat: dict[str, list[dict]] = {}
    for p in places:
        place_by_cat.setdefault(p["category"], []).append(p)

    all_categories = list(place_by_cat.keys())

    onboarding = []
    library_entries = []
    reviews = []
    place_likes = []
    place_dislikes = []
    checkins = []
    telemetry = []

    for user in users:
        persona = user["persona"]
        uid = user["user_id"]

        # Onboarding
        onboarding.append({"user_id": uid, "question_key": "preferred_categories", "answer_values": persona["preferred_categories"][:5]})
        onboarding.append({"user_id": uid, "question_key": "preferred_price_levels", "answer_values": [str(p) for p in persona["preferred_price"][:4]]})
        onboarding.append({"user_id": uid, "question_key": "preferred_vibes", "answer_values": persona["preferred_vibes"][:5]})
        onboarding.append({"user_id": uid, "question_key": "preferred_for", "answer_values": persona["preferred_for"][:4]})

        # Generate interactions
        n_interactions = max(5, int(rng.gauss(interactions_per_user, interactions_per_user * 0.3)))
        for _ in range(n_interactions):
            # Place selection: 60% preferred, 40% random discovery
            roll = rng.random()
            if roll < 0.60:
                cat = rng.choice(persona["preferred_categories"])
            else:
                cat = rng.choice(all_categories)

            candidates = place_by_cat.get(cat, [])
            if not candidates:
                candidates = places
            place = rng.choice(candidates)
            pid = place["public_id"]
            place_db_id = place.get("db_id", 0)

            # Timestamp: exponential distribution (more recent)
            age_hours = rng.expovariate(1 / (days * 12))
            ts = now - timedelta(hours=min(age_hours, days * 24))

            is_preferred = cat in persona["preferred_categories"]
            is_good = place["rating_average"] >= 3.5

            # Interaction type based on preference match
            if is_preferred and is_good:
                action_roll = rng.random()
                if action_roll < 0.30:
                    # Positive explicit: library entry
                    entry_type = rng.choice(["favorite", "want_to_go", "been_there"])
                    library_entries.append({"user_id": uid, "place_id": pid, "entry_type": entry_type, "created_at": ts})
                elif action_roll < 0.50:
                    # Positive: review 4-5
                    rating = rng.choice([4, 5])
                    reviews.append({"user_id": uid, "place_id": pid, "rating": rating, "created_at": ts})
                elif action_roll < 0.65:
                    # Like
                    place_likes.append({"user_id": uid, "place_id": pid, "place_db_id": place_db_id, "created_at": ts})
                elif action_roll < 0.75:
                    # Checkin
                    dist = rng.uniform(10, 500)
                    checkins.append({
                        "user_id": uid, "place_db_id": place_db_id,
                        "latitude": place["latitude"] + rng.gauss(0, 0.001),
                        "longitude": place["longitude"] + rng.gauss(0, 0.001),
                        "distance_meters": round(dist, 2),
                        "verified": dist <= 200,
                        "created_at": ts,
                    })
                else:
                    # Telemetry: click + detail view + view time
                    telemetry.append({"user_id": uid, "entity_id": pid, "event_type": "impression", "created_at": ts})
                    telemetry.append({"user_id": uid, "entity_id": pid, "event_type": "feed_position_click", "created_at": ts})
                    view_ms = max(500, int(rng.lognormvariate(math.log(20000), 0.8)))
                    telemetry.append({"user_id": uid, "entity_id": pid, "event_type": "view_time", "payload": {"duration_ms": view_ms}, "created_at": ts})
            else:
                # Non-preferred or low-rated: mostly impressions
                telemetry.append({"user_id": uid, "entity_id": pid, "event_type": "impression", "created_at": ts})
                if rng.random() < 0.15:
                    telemetry.append({"user_id": uid, "entity_id": pid, "event_type": "feed_position_click", "created_at": ts})
                    view_ms = max(500, int(rng.lognormvariate(math.log(3000), 1.0)))
                    telemetry.append({"user_id": uid, "entity_id": pid, "event_type": "view_time", "payload": {"duration_ms": view_ms}, "created_at": ts})
                if rng.random() < 0.05 and not is_good:
                    place_dislikes.append({"user_id": uid, "place_id": pid, "place_db_id": place_db_id, "created_at": ts})

    return {
        "onboarding": onboarding,
        "library_entries": library_entries,
        "reviews": reviews,
        "place_likes": place_likes,
        "place_dislikes": place_dislikes,
        "checkins": checkins,
        "telemetry": telemetry,
    }
