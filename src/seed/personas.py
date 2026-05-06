"""User personas for synthetic data generation."""

from __future__ import annotations

import random

CATEGORIES = [
    "restaurant", "bar", "cafe", "nightclub", "park", "museum",
    "cinema", "shopping", "fitness", "spa", "bakery", "ice_cream",
    "attraction", "theater", "beach",
]

HIGHLIGHTS = [
    "live_music", "quiet", "outdoor_seating", "pet_friendly",
    "family_friendly", "romantic", "group_friendly", "instagrammable",
    "craft_drinks", "local_food", "vegan_options", "late_night",
]

RECOMMENDED_FOR = [
    "date", "friends", "family", "solo", "work",
    "celebration", "casual", "exercise", "culture",
]

PERSONAS = [
    {
        "name": "foodie",
        "weight": 0.25,
        "preferred_categories": ["restaurant", "cafe", "bakery"],
        "preferred_price": [2, 3],
        "preferred_vibes": ["local_food", "craft_drinks", "instagrammable"],
        "preferred_for": ["date", "friends"],
    },
    {
        "name": "nightlife",
        "weight": 0.15,
        "preferred_categories": ["bar", "nightclub"],
        "preferred_price": [2, 3, 4],
        "preferred_vibes": ["live_music", "late_night", "group_friendly"],
        "preferred_for": ["friends", "celebration"],
    },
    {
        "name": "family",
        "weight": 0.20,
        "preferred_categories": ["restaurant", "park", "cinema", "museum"],
        "preferred_price": [1, 2],
        "preferred_vibes": ["family_friendly", "outdoor_seating", "quiet"],
        "preferred_for": ["family", "casual"],
    },
    {
        "name": "active",
        "weight": 0.15,
        "preferred_categories": ["park", "fitness", "beach"],
        "preferred_price": [1, 2],
        "preferred_vibes": ["outdoor_seating", "pet_friendly"],
        "preferred_for": ["exercise", "solo"],
    },
    {
        "name": "culture",
        "weight": 0.10,
        "preferred_categories": ["museum", "theater", "attraction"],
        "preferred_price": [2, 3],
        "preferred_vibes": ["quiet", "instagrammable"],
        "preferred_for": ["culture", "solo", "date"],
    },
    {
        "name": "explorer",
        "weight": 0.15,
        "preferred_categories": [],
        "preferred_price": [1, 2, 3],
        "preferred_vibes": [],
        "preferred_for": [],
    },
]


def pick_persona(rng: random.Random) -> dict:
    """Pick a persona weighted by distribution."""
    weights = [p["weight"] for p in PERSONAS]
    persona = rng.choices(PERSONAS, weights=weights, k=1)[0]
    if persona["name"] == "explorer":
        persona = dict(persona)  # copy to avoid mutating the template
        persona["preferred_categories"] = rng.sample(CATEGORIES, 5)
        persona["preferred_vibes"] = rng.sample(HIGHLIGHTS, 4)
        persona["preferred_for"] = rng.sample(RECOMMENDED_FOR, 3)
    return persona
