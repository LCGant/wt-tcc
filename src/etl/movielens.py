"""MovieLens / Last.fm loaders.

Maps public benchmarks to the schema used by the offline evaluator,
so the same pipeline can be evaluated against multiple real datasets.

Mapping:
  - user_id (int)  -> user_id (str: "ml_u{N}")
  - movie_id (int) -> place_id (str: "ml_m{N}")
  - rating (1-5):
      >= 4: weight = 1.0    (positive: interaction)
       == 3: weight = 0.0   (neutral; dropped by ETL filter)
       <= 2: weight = -0.5  (negative)
  - genres (19 binary cols) -> first non-zero genre name as `category`
  - unix timestamp -> timestamp (datetime)

All ids are string-prefixed with `ml_` to avoid clashing with the synthetic
catalog if both run in the same DB session.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def load_movielens_100k(root: Path) -> dict[str, pd.DataFrame]:
    """Load MovieLens-100K from a directory containing u.data and u.item.

    Returns a dict with keys:
        places_df       — schema-compatible with extract_place_features()
        signals_df      — schema-compatible with extract_all_signals()
        onboarding_df   — empty DataFrame (MovieLens has no onboarding)
    """
    udata = root / "u.data"
    uitem = root / "u.item"
    if not udata.exists() or not uitem.exists():
        raise FileNotFoundError(f"MovieLens files not found in {root}")

    # ── Items (movies → places) ──
    items = pd.read_csv(
        uitem, sep="|", header=None, encoding="latin-1",
        names=["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
        + [f"g_{g}" for g in GENRE_NAMES],
        usecols=["movie_id", "title", "release_date"] + [f"g_{g}" for g in GENRE_NAMES],
    )
    # Pick first non-zero genre as primary category
    def primary_genre(row):
        for g in GENRE_NAMES:
            if row.get(f"g_{g}", 0) == 1 and g != "unknown":
                return g
        return "unknown"
    items["category"] = items.apply(primary_genre, axis=1)
    items["public_id"] = items["movie_id"].apply(lambda x: f"ml_m{x}")

    # All non-zero genres → highlights (so TF-IDF document discriminates by genre combo)
    def all_genres(row):
        return [g for g in GENRE_NAMES if g != "unknown" and row.get(f"g_{g}", 0) == 1]
    items["highlights"] = items.apply(all_genres, axis=1)

    places_df = pd.DataFrame({
        "id": items["movie_id"].astype(int),
        "public_id": items["public_id"],
        "name": items["title"],
        "category": items["category"],
        "highlights": items["highlights"],
        "description": items["title"],  # used as text feature
        "address_line": "",
        "city": "",
        "country_code": "US",
        "latitude": 0.0,
        "longitude": 0.0,
        "rating_average": 0.0,
        "ratings_count": 0,
        "price_level": 0,
        "external_source": "movielens",
        "external_id": items["movie_id"].astype(str),
        "verified": False,
        "status": "active",
        "is_accessible": False,
        "is_outdoor": False,
        "is_family_friendly": False,
        "is_pet_friendly": False,
        "has_parking": False,
        "has_wifi": False,
        "serves_alcohol": False,
        "accepts_reservations": False,
        "adult_content": False,
    })

    # ── Ratings (→ signals) ──
    ratings = pd.read_csv(
        udata, sep="\t", header=None,
        names=["user_id", "movie_id", "rating", "ts_unix"],
    )

    # Map rating → signal_type and weight
    def rating_to_weight(r: int) -> float:
        if r >= 4:
            return 1.0
        if r == 3:
            return 0.0
        return -0.5

    def rating_to_signal_type(r: int) -> str:
        if r >= 4:
            return "ml_review_positive"
        if r == 3:
            return "ml_review_neutral"
        return "ml_review_negative"

    ratings["weight"] = ratings["rating"].apply(rating_to_weight)
    ratings["signal_type"] = ratings["rating"].apply(rating_to_signal_type)
    ratings = ratings[ratings["weight"] != 0.0]  # drop neutral

    ratings["timestamp"] = pd.to_datetime(ratings["ts_unix"], unit="s", utc=True)
    ratings["user_id"] = ratings["user_id"].apply(lambda x: f"ml_u{x}")
    ratings["place_id"] = ratings["movie_id"].apply(lambda x: f"ml_m{x}")

    signals_df = ratings[["user_id", "place_id", "signal_type", "weight", "timestamp"]].copy()

    # ── Onboarding (empty) ──
    onboarding_df = pd.DataFrame(columns=["user_id", "actor_id", "question_key", "answer_values"])

    return {
        "places_df": places_df,
        "signals_df": signals_df.reset_index(drop=True),
        "onboarding_df": onboarding_df,
    }


def load_movielens_1m(root: Path) -> dict[str, pd.DataFrame]:
    """Load MovieLens-1M from a directory containing ratings.dat and movies.dat.

    Format differs from 100K:
      - separator is '::' instead of tab/pipe
      - genres are pipe-separated in a single column instead of binary columns
      - has 1M ratings, 6040 users, 3883 movies (~25× larger than 100K)

    Same return contract as load_movielens_100k.
    """
    ratings_path = root / "ratings.dat"
    movies_path = root / "movies.dat"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(f"MovieLens-1M files not found in {root}")

    # ── Items (movies → places) ──
    items = pd.read_csv(
        movies_path, sep="::", header=None, engine="python", encoding="latin-1",
        names=["movie_id", "title", "genres"],
    )

    def primary_genre(genres_str: str) -> str:
        if not isinstance(genres_str, str) or not genres_str:
            return "unknown"
        return genres_str.split("|")[0]

    items["category"] = items["genres"].apply(primary_genre)
    items["public_id"] = items["movie_id"].apply(lambda x: f"ml1m_m{x}")

    # All genres → highlights (so TF-IDF document discriminates by genre combo)
    def all_genres(genres_str: str) -> list:
        if not isinstance(genres_str, str) or not genres_str:
            return []
        return genres_str.split("|")
    items["highlights"] = items["genres"].apply(all_genres)

    places_df = pd.DataFrame({
        "id": items["movie_id"].astype(int),
        "public_id": items["public_id"],
        "name": items["title"],
        "category": items["category"],
        "highlights": items["highlights"],
        "description": items["title"],
        "address_line": "",
        "city": "",
        "country_code": "US",
        "latitude": 0.0,
        "longitude": 0.0,
        "rating_average": 0.0,
        "ratings_count": 0,
        "price_level": 0,
        "external_source": "movielens-1m",
        "external_id": items["movie_id"].astype(str),
        "verified": False,
        "status": "active",
        "is_accessible": False,
        "is_outdoor": False,
        "is_family_friendly": False,
        "is_pet_friendly": False,
        "has_parking": False,
        "has_wifi": False,
        "serves_alcohol": False,
        "accepts_reservations": False,
        "adult_content": False,
    })

    # ── Ratings (→ signals) ──
    ratings = pd.read_csv(
        ratings_path, sep="::", header=None, engine="python",
        names=["user_id", "movie_id", "rating", "ts_unix"],
    )

    def rating_to_weight(r: int) -> float:
        if r >= 4:
            return 1.0
        if r == 3:
            return 0.0
        return -0.5

    def rating_to_signal_type(r: int) -> str:
        if r >= 4:
            return "ml_review_positive"
        if r == 3:
            return "ml_review_neutral"
        return "ml_review_negative"

    ratings["weight"] = ratings["rating"].apply(rating_to_weight)
    ratings["signal_type"] = ratings["rating"].apply(rating_to_signal_type)
    ratings = ratings[ratings["weight"] != 0.0]

    ratings["timestamp"] = pd.to_datetime(ratings["ts_unix"], unit="s", utc=True)
    ratings["user_id"] = ratings["user_id"].apply(lambda x: f"ml1m_u{x}")
    ratings["place_id"] = ratings["movie_id"].apply(lambda x: f"ml1m_m{x}")

    signals_df = ratings[["user_id", "place_id", "signal_type", "weight", "timestamp"]].copy()

    onboarding_df = pd.DataFrame(columns=["user_id", "actor_id", "question_key", "answer_values"])

    return {
        "places_df": places_df,
        "signals_df": signals_df.reset_index(drop=True),
        "onboarding_df": onboarding_df,
    }


def load_lastfm_2k(root: Path, top_n_artists: int = 1500) -> dict[str, pd.DataFrame]:
    """Load HetRec-2011 Last.fm-2k dataset.

    Files used:
      - artists.dat:                       id, name, url, pictureURL
      - user_taggedartists-timestamps.dat: userID, artistID, tagID, timestamp (ms)
      - tags.dat:                          tagID, tagValue (used as feature/category)

    Mapping:
      - userID  -> user_id  (str: "lf_u{N}")
      - artistID -> place_id (str: "lf_a{N}")
      - Each (user, artist) pair: weight ∈ (0, 1] saturated by log of n_tags
      - timestamp (ms) -> datetime (taking earliest tag time per user-artist pair)
      - Most-frequent tag for an artist -> primary `category`
      - All tags applied to artist -> `highlights` (so TF-IDF discriminates)

    Subsampling: keep only the top `top_n_artists` artists by tagging volume,
    to stay within the architecture's tested operating regime (~1500-2000 items).
    """
    import math
    artists_path = root / "artists.dat"
    tags_path = root / "tags.dat"
    events_path = root / "user_taggedartists-timestamps.dat"
    for p in (artists_path, tags_path, events_path):
        if not p.exists():
            raise FileNotFoundError(f"Last.fm-2k file not found: {p}")

    tags = pd.read_csv(tags_path, sep="\t", encoding="latin-1")
    tag_map = dict(zip(tags["tagID"].astype(int), tags["tagValue"].astype(str)))

    events = pd.read_csv(events_path, sep="\t")
    events = events[events["timestamp"] > 0]  # drop invalid timestamps

    # Subsample to top-N artists by tagging volume
    artist_volume = events.groupby("artistID").size().sort_values(ascending=False)
    top_artist_ids = set(artist_volume.head(top_n_artists).index)
    events = events[events["artistID"].isin(top_artist_ids)]

    def _slug(s: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in str(s).lower())[:32]

    artist_tags: dict = {}
    for aid, sub in events.groupby("artistID"):
        tag_counts = sub["tagID"].value_counts()
        ranked = [_slug(tag_map.get(int(t), "")) for t, _ in tag_counts.items()]
        ranked = [r for r in ranked if r]
        artist_tags[aid] = ranked[:10]

    artists = pd.read_csv(
        artists_path, sep="\t", encoding="latin-1", usecols=["id", "name"],
    )
    artists = artists[artists["id"].isin(top_artist_ids)].copy()
    artists["public_id"] = artists["id"].apply(lambda x: f"lf_a{x}")
    artists["category"] = artists["id"].apply(
        lambda x: artist_tags.get(x, ["unknown"])[0] if artist_tags.get(x) else "unknown"
    )
    artists["highlights"] = artists["id"].apply(lambda x: artist_tags.get(x, []))

    places_df = pd.DataFrame({
        "id": artists["id"].astype(int),
        "public_id": artists["public_id"],
        "name": artists["name"],
        "category": artists["category"],
        "highlights": artists["highlights"],
        "description": artists["name"],
        "address_line": "",
        "city": "",
        "country_code": "XX",
        "latitude": 0.0,
        "longitude": 0.0,
        "rating_average": 0.0,
        "ratings_count": 0,
        "price_level": 0,
        "external_source": "lastfm-2k",
        "external_id": artists["id"].astype(str),
        "verified": False,
        "status": "active",
        "is_accessible": False,
        "is_outdoor": False,
        "is_family_friendly": False,
        "is_pet_friendly": False,
        "has_parking": False,
        "has_wifi": False,
        "serves_alcohol": False,
        "accepts_reservations": False,
        "adult_content": False,
    })

    # Per-pair signals
    sig = events.groupby(["userID", "artistID"]).agg(
        timestamp=("timestamp", "min"),
        n_tags=("tagID", "count"),
    ).reset_index()
    sig["weight"] = sig["n_tags"].apply(lambda n: min(1.0, math.log(1 + n) / math.log(5)))
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], unit="ms", utc=True)
    sig["user_id"] = sig["userID"].apply(lambda x: f"lf_u{x}")
    sig["place_id"] = sig["artistID"].apply(lambda x: f"lf_a{x}")
    sig["signal_type"] = "lf_tag_positive"

    signals_df = sig[["user_id", "place_id", "signal_type", "weight", "timestamp"]].copy()
    onboarding_df = pd.DataFrame(columns=["user_id", "actor_id", "question_key", "answer_values"])

    return {
        "places_df": places_df,
        "signals_df": signals_df.reset_index(drop=True),
        "onboarding_df": onboarding_df,
    }
