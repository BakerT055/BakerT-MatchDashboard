"""Loading and cleaning results / fixtures files.

Works directly with football-data.co.uk files (single CSVs or the all-euro-data-YYYY-YYYY.xlsx
workbooks with one sheet per league) as well as simple generic files (Date, League, HomeTeam,
AwayTeam, FTHG, FTAG, HTHG, HTAG, Home odds, ...).  No row limit.
"""
from __future__ import annotations
import json
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import COLUMN_ALIASES, ODDS_SOURCES, DATA_DIR, ALIASES_FILE

NUM_COLS = ["FTHG", "FTAG", "HTHG", "HTAG"]


def _norm(c) -> str:
    return re.sub(r"[\s_]+", "", str(c).strip().lower())


def load_team_aliases() -> dict:
    if ALIASES_FILE.exists():
        try:
            raw = json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
            return {str(k).strip(): str(v).strip() for k, v in raw.items() if not str(k).startswith("_")}
        except Exception:
            return {}
    return {}


def season_label(d: pd.Series) -> pd.Series:
    y = d.dt.year
    start = np.where(d.dt.month >= 7, y, y - 1)
    out = pd.Series([f"{int(a)}/{int(a) + 1}" if a == a else None for a in start], index=d.index)
    return out.where(d.notna())


def _parse_dates(col: pd.Series) -> pd.Series:
    """ISO dates (2026-11-07) are read year-first; everything else (07/11/2026, 07/11/26) day-first,
    which is the football-data.co.uk convention."""
    txt = col.astype(str).str.strip()
    iso = txt.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    out = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(txt[iso].str.slice(0, 10), errors="coerce", format="%Y-%m-%d")
    if (~iso).any():
        out[~iso] = pd.to_datetime(txt[~iso], dayfirst=True, errors="coerce", format="mixed")
    return out


def normalize(df: pd.DataFrame, default_league: str = "ALL", aliases: dict | None = None) -> pd.DataFrame:
    """Return a frame with canonical columns Date, League, HomeTeam, AwayTeam, FTHG, FTAG, HTHG, HTAG,
    Season, plus one column per odds field and source, named '<field>__<source>' (e.g. oh__B365)."""
    cmap = {_norm(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for canon, names in COLUMN_ALIASES.items():
        src = next((n for n in [_norm(canon)] + names if n in cmap), None)
        out[canon] = df[cmap[src]] if src else np.nan
    if out["League"].isna().all():
        out["League"] = default_league
    out["League"] = out["League"].fillna(default_league).astype(str).str.strip()

    if not pd.api.types.is_datetime64_any_dtype(out["Date"]):
        out["Date"] = _parse_dates(out["Date"])
    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    aliases = aliases or {}
    for c in ("HomeTeam", "AwayTeam"):
        out[c] = out[c].astype("object").where(out[c].notna(), None)
        out[c] = out[c].map(lambda v: None if v is None or str(v).strip() in ("", "nan") else aliases.get(str(v).strip(), str(v).strip()))
    out = out.dropna(subset=["HomeTeam", "AwayTeam"])
    out["Season"] = season_label(out["Date"])

    for field, sources in ODDS_SOURCES.items():
        for label, col in sources.items():
            if col in cmap:
                out[f"{field}__{label}"] = pd.to_numeric(df.loc[out.index, cmap[col]], errors="coerce")
    return out


def resolve_odds(df: pd.DataFrame, prefs: Iterable[str]) -> pd.DataFrame:
    """Add oh/od/oa/oo/ou/by/bn columns, taking the first available source in `prefs` per row."""
    df = df.copy()
    for field in ODDS_SOURCES:
        s = pd.Series(np.nan, index=df.index, dtype="float64")
        for label in prefs:
            col = f"{field}__{label}"
            if col in df.columns:
                s = s.combine_first(df[col])
        df[field] = s.where(s > 1.0)
    return df


def _read_any(f, name: str) -> list[tuple[str, pd.DataFrame]]:
    ext = Path(name).suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(f, sheet_name=None)
        return [(str(k), v) for k, v in sheets.items()]
    for enc in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(f, "seek"):
                f.seek(0)
            return [(Path(name).stem, pd.read_csv(f, encoding=enc, on_bad_lines="skip"))]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {name}")


def load_files(files: list, aliases: dict | None = None) -> pd.DataFrame:
    """files: list of paths or file-like objects with a .name (e.g. Streamlit uploads)."""
    aliases = aliases if aliases is not None else load_team_aliases()
    parts = []
    for f in files:
        name = getattr(f, "name", str(f))
        src = f if hasattr(f, "read") else str(f)
        for sheet, raw in _read_any(src, name):
            if raw.empty:
                continue
            n = normalize(raw, default_league=sheet, aliases=aliases)
            if len(n):
                parts.append(n)
    if not parts:
        return pd.DataFrame(columns=["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG", "Season"])
    df = pd.concat(parts)
    return df.reset_index(drop=True)


def load_history(files: list, aliases: dict | None = None) -> pd.DataFrame:
    df = load_files(files, aliases)
    df = df.dropna(subset=["FTHG", "FTAG"])
    df = df.drop_duplicates(subset=["League", "Date", "HomeTeam", "AwayTeam"])
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def load_folder(folder: str, aliases: dict | None = None) -> pd.DataFrame:
    p = Path(folder)
    files = sorted([f for f in p.glob("*") if f.suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xls")])
    return load_history(files, aliases) if files else pd.DataFrame()


def load_fixtures(files: list, aliases: dict | None = None) -> pd.DataFrame:
    """Fixtures keep their original index so predictions can be appended to the raw file later."""
    df = load_files(files, aliases)
    return df.sort_values("Date", kind="stable", na_position="last").reset_index(drop=True) if len(df) else df


# ---------------- persistence ----------------
def save_dataset(df: pd.DataFrame, name: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    p = DATA_DIR / f"{name}.pkl"
    df.to_pickle(p)
    return p


def load_saved(name: str) -> pd.DataFrame | None:
    p = DATA_DIR / f"{name}.pkl"
    if p.exists():
        try:
            return pd.read_pickle(p)
        except Exception:
            return None
    return None


# ---------------- diagnostics ----------------
def summarize(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"rows": 0}
    return {
        "rows": len(df),
        "leagues": df["League"].nunique(),
        "teams": pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]])).size,
        "from": df["Date"].min(), "to": df["Date"].max(),
        "seasons": df["Season"].nunique() if "Season" in df else 0,
        "with_ht": int((df["HTHG"].notna() & df["HTAG"].notna()).sum()),
        "odds_sources": sorted({c.split("__")[1] for c in df.columns if "__" in c and df[c].notna().any()}),
    }


def check_team_names(fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Fixture teams that never appear in History for that league, with close-match suggestions."""
    rows = []
    for lg, g in fixtures.groupby("League"):
        hg = history[history["League"] == lg]
        known = set(hg["HomeTeam"]) | set(hg["AwayTeam"])
        if not known:
            known = set(history["HomeTeam"]) | set(history["AwayTeam"])
            note = "league not in History"
        else:
            note = ""
        for t in sorted(set(g["HomeTeam"]) | set(g["AwayTeam"])):
            if t not in known:
                sug = get_close_matches(t, list(known), n=3, cutoff=0.6)
                rows.append({"League": lg, "Fixture team": t, "Suggestions": ", ".join(sug), "Note": note})
    return pd.DataFrame(rows, columns=["League", "Fixture team", "Suggestions", "Note"])
