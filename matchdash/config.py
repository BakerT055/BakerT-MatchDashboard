"""Everything you are likely to want to customise lives here or in settings.json.

* Settings        – model/UI parameters (edit in the app sidebar, saved to settings.json)
* TIERS           – colour tiers used to flag strong / weak predictions
* COLUMN_ALIASES  – how column names in your files are recognised
* ODDS_SOURCES    – which odds columns are read (football-data.co.uk names + generic names)
* PANELS          – how markets are grouped on the dashboard
Markets themselves are defined in markets.py.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SETTINGS_FILE = ROOT / "settings.json"
ALIASES_FILE = DATA_DIR / "team_aliases.json"

# name, background, text colour  (index 0 = strongest)
TIERS = [
    ("Very strong", "#1E7B34", "#FFFFFF"),
    ("Strong",      "#C6E7C1", "#1B5E20"),
    ("Lean",        "#FFF2B3", "#7A5C00"),
    ("Weak",        "#FAD7B0", "#8A4B08"),
    ("Very weak",   "#F4C7C3", "#9C1C1C"),
]

# canonical column -> accepted names (compared lower-case, without spaces/underscores)
COLUMN_ALIASES = {
    "Date":     ["date", "matchdate", "kickoff"],
    "League":   ["div", "league", "competition", "lg"],
    "HomeTeam": ["hometeam", "home", "hometeamname", "team1"],
    "AwayTeam": ["awayteam", "away", "awayteamname", "team2"],
    "FTHG":     ["fthg", "hg", "homegoals", "fthomegoals"],
    "FTAG":     ["ftag", "ag", "awaygoals", "ftawaygoals"],
    "HTHG":     ["hthg", "hthomegoals"],
    "HTAG":     ["htag", "htawaygoals"],
}

# odds field -> {source label: normalised column name}.  oh/od/oa = 1X2, oo/ou = Over/Under 2.5, by/bn = BTTS yes/no
ODDS_SOURCES = {
    "oh": {"B365": "b365h", "PS": "psh", "Avg": "avgh", "Max": "maxh", "Generic": "homeodds"},
    "od": {"B365": "b365d", "PS": "psd", "Avg": "avgd", "Max": "maxd", "Generic": "drawodds"},
    "oa": {"B365": "b365a", "PS": "psa", "Avg": "avga", "Max": "maxa", "Generic": "awayodds"},
    "oo": {"B365": "b365>2.5", "PS": "p>2.5", "Avg": "avg>2.5", "Max": "max>2.5", "Generic": "over2.5odds"},
    "ou": {"B365": "b365<2.5", "PS": "p<2.5", "Avg": "avg<2.5", "Max": "max<2.5", "Generic": "under2.5odds"},
    "by": {"Generic": "bttsyesodds"},
    "bn": {"Generic": "bttsnoodds"},
}
ODDS_LABELS = {"oh": "Home odds", "od": "Draw odds", "oa": "Away odds", "oo": "Over 2.5 odds",
               "ou": "Under 2.5 odds", "by": "BTTS Yes odds", "bn": "BTTS No odds"}
ODDS_PREF_CHOICES = ["B365", "PS", "Avg", "Max", "Generic"]

# dashboard layout: panel title -> market groups (see markets.py)
PANELS = [
    ("Result & half-time", ["Result", "Double chance", "DNB", "Half-time"]),
    ("Goal markets",       ["Goals", "BTTS", "Team goals"]),
    ("HT/FT combinations", ["HT/FT"]),
]


@dataclass
class Settings:
    # --- model ---
    shrinkage: float = 5.0            # pseudo-games pulling small samples to the league average
    history_from: str = ""            # ISO date, "" = all history (team-form model only; H2H always uses all)
    half_life_days: float = 0.0       # recency weighting; 0 = off (all matches equal)
    dc_rho: float = 0.0               # Dixon-Coles low-score correlation; 0 = plain Poisson
    only_fixture_league: bool = True  # use only the fixture's league for averages
    max_goals: int = 10               # goals per team in the scoreline grid
    ht_share_fallback: float = 0.45   # 1st-half share of goals when no HT data
    # --- warnings / value ---
    min_games_warn: int = 8
    min_matches_warn: int = 30
    edge_min: float = 0.05
    # --- colour tiers (probability thresholds) ---
    tier_very_strong: float = 0.70
    tier_strong: float = 0.55
    tier_lean: float = 0.45
    tier_weak: float = 0.30
    # --- misc ---
    odds_preference: list = field(default_factory=lambda: ["B365", "PS", "Avg", "Max", "Generic"])
    h2h_rows: int = 8
    # --- engine ---
    engine: str = "Poisson"           # Poisson | ML | Blend
    ml_blend: float = 0.5             # ML weight when engine = Blend

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                for f in fields(cls):
                    if f.name in raw:
                        setattr(s, f.name, raw[f.name])
            except Exception:
                pass
        return s

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def model_key(self) -> str:
        """Settings that change the fitted model (used for caching)."""
        d = asdict(self)
        for k in ("min_games_warn", "min_matches_warn", "edge_min", "tier_very_strong", "tier_strong",
                  "tier_lean", "tier_weak", "odds_preference", "h2h_rows", "engine", "ml_blend"):
            d.pop(k, None)
        return json.dumps(d, sort_keys=True)
