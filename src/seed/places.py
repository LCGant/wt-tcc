"""Generate synthetic places."""

from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timedelta, timezone

from src.seed.personas import HIGHLIGHTS, RECOMMENDED_FOR

# Category popularity distribution (sums to ~1.0)
_CATEGORY_WEIGHTS = {
    "restaurant": 0.25, "bar": 0.15, "cafe": 0.12, "park": 0.08,
    "museum": 0.05, "nightclub": 0.05, "cinema": 0.04, "shopping": 0.04,
    "fitness": 0.04, "bakery": 0.04, "ice_cream": 0.03, "spa": 0.03,
    "attraction": 0.03, "theater": 0.02, "beach": 0.03,
}


def generate_places(
    n: int,
    center_lat: float,
    center_lng: float,
    radius_km: float,
    rng: random.Random,
) -> list[dict]:
    """Generate N synthetic places with realistic distributions."""
    categories = list(_CATEGORY_WEIGHTS.keys())
    cat_weights = [_CATEGORY_WEIGHTS[c] for c in categories]
    now = datetime.now(timezone.utc)

    places = []
    for _ in range(n):
        cat = rng.choices(categories, weights=cat_weights, k=1)[0]

        # Random point within radius
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0, radius_km)
        lat = center_lat + (dist / 111.32) * math.cos(angle)
        lng = center_lng + (dist / (111.32 * math.cos(math.radians(center_lat)))) * math.sin(angle)

        # Rating: Normal(3.8, 0.7), clamp [1, 5]
        rating = max(1.0, min(5.0, rng.gauss(3.8, 0.7)))
        rating_count = max(1, int(rng.expovariate(1 / 30)))

        # Price: exponential distribution
        price = rng.choices([0, 1, 2, 3, 4], weights=[5, 40, 30, 20, 5], k=1)[0]

        # Highlights: 2-5 random
        n_highlights = rng.randint(2, 5)
        highlights = rng.sample(HIGHLIGHTS, min(n_highlights, len(HIGHLIGHTS)))

        # Recommended for: 1-3 random
        n_rec = rng.randint(1, 3)
        recommended = rng.sample(RECOMMENDED_FOR, min(n_rec, len(RECOMMENDED_FOR)))

        # Age: 1-365 days
        age_days = rng.randint(1, 365)

        places.append({
            "public_id": str(uuid.uuid4()),
            "name": f"{cat.title()} {rng.randint(1, 9999)}",
            "category": cat,
            "latitude": round(lat, 6),
            "longitude": round(lng, 6),
            "rating_average": round(rating, 2),
            "ratings_count": rating_count,
            "price_level": price,
            "verified": rng.random() < 0.2,
            "highlights": highlights,
            "recommended_for": recommended,
            "created_at": now - timedelta(days=age_days),
        })

    return places
