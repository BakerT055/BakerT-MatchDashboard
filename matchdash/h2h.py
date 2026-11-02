"""Head-to-head and 'actual results' helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import resolve_odds


def _valid(df):
    return df[df["FTHG"].notna() & df["FTAG"].notna()]


def meetings(history: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    m = history[((history["HomeTeam"] == home) & (history["AwayTeam"] == away)) |
                ((history["HomeTeam"] == away) & (history["AwayTeam"] == home))]
    return _valid(m).sort_values("Date", ascending=False, kind="stable")


def _profile(gf: pd.Series, ga: pd.Series) -> dict:
    n = len(gf)
    if n == 0:
        return {"Games": 0, "Win %": np.nan, "Draw %": np.nan, "Over 2.5 %": np.nan, "BTTS %": np.nan,
                "Avg goals scored": np.nan, "Avg goals conceded": np.nan}
    return {"Games": n, "Win %": float((gf > ga).mean()), "Draw %": float((gf == ga).mean()),
            "Over 2.5 %": float(((gf + ga) > 2.5).mean()), "BTTS %": float(((gf > 0) & (ga > 0)).mean()),
            "Avg goals scored": float(gf.mean()), "Avg goals conceded": float(ga.mean())}


def history_check(snap, history: pd.DataFrame, home: str, away: str, league) -> pd.DataFrame:
    d = snap.league_data(league)
    h = d[d["HomeTeam"] == home]
    a = d[d["AwayTeam"] == away]
    m = meetings(history, home, away)
    gf = np.where(m["HomeTeam"] == home, m["FTHG"], m["FTAG"]); ga = np.where(m["HomeTeam"] == home, m["FTAG"], m["FTHG"])
    cols = {"Home @ home": _profile(h["FTHG"], h["FTAG"]),
            "Away @ away": _profile(a["FTAG"], a["FTHG"]),
            "H2H (all history)": _profile(pd.Series(gf, dtype=float), pd.Series(ga, dtype=float))}
    return pd.DataFrame(cols)


def recent_table(history: pd.DataFrame, home: str, away: str, n: int, prefs) -> pd.DataFrame:
    m = resolve_odds(meetings(history, home, away).head(n), prefs)
    out = pd.DataFrame({
        "Date": m["Date"].dt.strftime("%d-%b-%Y"),
        "Home": m["HomeTeam"], "Away": m["AwayTeam"],
        "FT": m["FTHG"].astype(int).astype(str) + "-" + m["FTAG"].astype(int).astype(str),
        "HT": np.where(m["HTHG"].notna() & m["HTAG"].notna(),
                       m["HTHG"].fillna(0).astype(int).astype(str) + "-" + m["HTAG"].fillna(0).astype(int).astype(str), "–"),
        "Goals": (m["FTHG"] + m["FTAG"]).astype(int),
        "O/U 2.5": np.where(m["FTHG"] + m["FTAG"] > 2.5, "Over 2.5", "Under 2.5"),
        "BTTS": np.where((m["FTHG"] > 0) & (m["FTAG"] > 0), "BTTS", "No BTTS"),
        "1": m["oh"], "X": m["od"], "2": m["oa"], "O2.5": m["oo"], "U2.5": m["ou"],
    })
    return out.reset_index(drop=True)
