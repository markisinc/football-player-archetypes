"""Player style archetypes and similar-player search on StatsBomb Open Data 2015/16
(Premier League, La Liga, Serie A, Ligue 1).

Ported from profiling.py (2012/13 Opta file). Same pipeline shape:
per-match player features -> player-season rates -> impute/cap/scale -> PCA -> clusters + neighbours,
with whole-match bootstrap validation. Changes:
  * minutes come from StatsBomb lineups (exact on/off clock times), no inference
  * style features only in the model; success rates and output (goals, xG) are context
  * defensive counts are possession-adjusted (a team with less of the ball defends more)
  * two-level archetypes: position group (from minutes by StatsBomb position) -> sub-roles found within each group
  * neighbour stability: how often each neighbour stays in the top 5 across match resamples

Run: python src/sb_profiling.py [--bootstraps 30] [--min-minutes 900]
Needs data/statsbomb/*.parquet from src/sb_load.py. Writes outputs/profiling_sb/ and docs/profiles.json.
"""
from pathlib import Path
import argparse
import json
import warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sb_load import read_events, read_lineups

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "statsbomb"
OUT = ROOT / "outputs" / "profiling_sb"
WORK = ROOT / "work" / "profiling_sb"

# ---------------------------------------------------------------- definitions
GROUPS = {
    "Goalkeeper": "Goalkeeper",
    "Center Back": "Centre-back", "Left Center Back": "Centre-back", "Right Center Back": "Centre-back",
    "Left Back": "Full-back", "Right Back": "Full-back", "Left Wing Back": "Full-back", "Right Wing Back": "Full-back",
    "Center Defensive Midfield": "Midfielder", "Left Defensive Midfield": "Midfielder", "Right Defensive Midfield": "Midfielder",
    "Center Midfield": "Midfielder", "Left Center Midfield": "Midfielder", "Right Center Midfield": "Midfielder",
    "Center Attacking Midfield": "Attacking mid / winger", "Left Attacking Midfield": "Attacking mid / winger",
    "Right Attacking Midfield": "Attacking mid / winger", "Left Midfield": "Attacking mid / winger",
    "Right Midfield": "Attacking mid / winger", "Left Wing": "Attacking mid / winger", "Right Wing": "Attacking mid / winger",
    "Center Forward": "Forward", "Left Center Forward": "Forward", "Right Center Forward": "Forward", "Secondary Striker": "Forward",
}
GROUP_ORDER = ["Centre-back", "Full-back", "Midfielder", "Attacking mid / winger", "Forward"]

# key: (label, group). All are style inputs; context metrics are listed separately.
FEATURES = {
    "passes90": ("Passes", "Volume"),
    "receptions90": ("Receptions", "Volume"),
    "carries90": ("Carries", "Volume"),
    "dribbles90": ("Take-ons attempted", "Volume"),
    "long_share": ("Long-pass share", "Passing style"),
    "forward_share": ("Forward-pass share", "Passing style"),
    "prog_passes90": ("Progressive passes", "Passing style"),
    "crosses90": ("Crosses (open play)", "Passing style"),
    "shot_assists90": ("Passes leading to shots", "Passing style"),
    "prog_carries90": ("Progressive carries", "Carrying"),
    "avg_x": ("Average action depth", "Location"),
    "final_third": ("Final-third share", "Location"),
    "opp_box": ("Opponent-box share", "Location"),
    "own_box": ("Own-box share", "Location"),
    "width": ("Distance from centre", "Location"),
    "spread_y": ("Lateral spread", "Location"),
    "pressures90": ("Pressures (adj.)", "Defending"),
    "tackles90": ("Tackles (adj.)", "Defending"),
    "interceptions90": ("Interceptions (adj.)", "Defending"),
    "recoveries90": ("Ball recoveries (adj.)", "Defending"),
    "clearances90": ("Clearances (adj.)", "Defending"),
    "blocks90": ("Blocks (adj.)", "Defending"),
    "aerials90": ("Aerial duels", "Defending"),
    "def_height": ("Defensive action height", "Defending"),
    "fouls90": ("Fouls committed", "Defending"),
    "shots90": ("Shots", "Attacking"),
    "npxg_per_shot": ("Chance quality per shot", "Attacking"),
}
CONTEXT = {
    "pass_pct": "Pass completion %", "dribble_pct": "Take-on success %", "aerial_pct": "Aerial win %",
    "tackle_pct": "Tackle win %", "goals": "Goals (non-own)", "npxg90": "Non-penalty xG per 90",
    "dispossessed90": "Dispossessed per 90", "miscontrols90": "Miscontrols per 90", "team_possession": "Team possession %",
}
RADAR = ["passes90", "prog_passes90", "prog_carries90", "dribbles90", "shots90", "pressures90", "tackles90", "aerials90"]
ONBALL = ["Ball Receipt*", "Pass", "Dribble", "Shot", "Ball Recovery", "Interception", "Clearance", "Block", "Duel",
          "Miscontrol", "Dispossessed"]
DEFENSIVE = ["Pressure", "Interception", "Ball Recovery", "Block", "Clearance", "Foul Committed"]
SET_PIECES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}


def clean_json(v):
    if isinstance(v, dict):
        return {str(k): clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean_json(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 5) if np.isfinite(v) else None
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def clock(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    m, sec = str(s).split(":")
    return int(m) * 60 + int(sec)


# ---------------------------------------------------------------- minutes
def period_lengths(events):
    """Seconds played in each period, from the last event of the period (includes stoppage)."""
    e = events[events.period.isin([1, 2])]
    t = e.minute * 60 + e.second - np.where(e.period == 2, 2700, 0)
    return pd.DataFrame({"match_id": e.match_id, "period": e.period, "t": t}).groupby(["match_id", "period"]).t.max().unstack()


def reconstruct_minutes(lineups, lengths):
    """Elapsed minutes per player spell from lineup clock times. Halftime break excluded, stoppage included."""
    lu = lineups.copy()
    L1 = lu.match_id.map(lengths[1]); L2 = lu.match_id.map(lengths[2])
    def elapsed(clk, per, default):
        c = clk.map(clock).astype(float)
        out = np.where(per == 1, np.minimum(c, L1), L1 + np.clip(c - 2700, 0, None))
        out = np.where(c.isna(), default, out)
        return out.astype(float)
    total = (L1 + L2).values
    lu["start"] = elapsed(lu["from_clock"], lu["from_period"], 0.0) / 60
    lu["end"] = elapsed(lu["to_clock"], lu["to_period"].fillna(2), total) / 60
    lu["end"] = np.maximum(lu.end, lu.start)
    # Data quirk: a few lineups give a player a new spell after his own substitution/sending-off.
    # Cut anything that starts at or after the player's first permanent exit.
    exit_reason = lu.end_reason.fillna("").str.contains("Substitution - Off|Sent Off|Red Card", regex=True)
    first_exit = lu[exit_reason].groupby(["match_id", "player_id"]).end.min()
    fx = pd.Series(list(zip(lu.match_id, lu.player_id))).map(first_exit).values
    lu["ghost"] = ~np.isnan(fx.astype(float)) & (lu.start >= fx.astype(float) - 1e-9) & ~exit_reason & (lu.end > lu.start)
    lu.loc[lu.ghost, "end"] = lu.loc[lu.ghost, "start"]
    # ...and a spell that began before that exit cannot run past it (e.g. "Player On" after treatment, then subbed).
    fxf = fx.astype(float)
    over = ~exit_reason & ~np.isnan(fxf) & (lu.start < fxf) & (lu.end > fxf)
    lu.loc[over, "end"] = fxf[over.values]
    lu["minutes"] = lu.end - lu.start
    lu["group"] = lu.position.map(GROUPS).fillna("Unknown")
    # Starter = on the pitch at 0:00 (some lineups label starters "Tactical Shift" instead of "Starting XI").
    lu["starter"] = (lu.start == 0) & (lu["from_period"] == 1) & (lu.end > 0)
    lu["sent_off"] = lu.end_reason.fillna("").str.contains("Sent Off|Red Card", regex=True)
    # audit every team-match: 11 starters, plausible active counts at every change point
    audits = []
    for (m, t), side in lu.groupby(["match_id", "team_id"]):
        pts = np.unique(np.r_[side.start.values, side.end.values])
        counts = [((side.start <= (a + b) / 2) & ((a + b) / 2 < side.end)).sum() for a, b in zip(pts, pts[1:]) if b - a > 1e-6]
        starters = side[side.starter].player_id.nunique()
        audits.append(dict(match_id=m, team_id=t, starters=starters, min_active=min(counts), max_active=max(counts),
                           roster_issue=bool(starters != 11 or max(counts) > 11 or min(counts) < 7)))
    return lu, pd.DataFrame(audits)


def union_minutes(spells):
    """Minutes per player-match as the union of their spells, so overlapping lineup spells never double-count."""
    rows = []
    for (m, t, p), s in spells[spells.end > spells.start].groupby(["match_id", "team_id", "player_id"]):
        total, cur_s, cur_e = 0.0, None, None
        for a, b in sorted(zip(s.start, s.end)):
            if cur_e is None or a > cur_e:
                if cur_e is not None:
                    total += cur_e - cur_s
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        total += cur_e - cur_s
        rows.append((m, t, p, total))
    return pd.DataFrame(rows, columns=["match_id", "team_id", "player_id", "minutes"])


# ---------------------------------------------------------------- per-match features
def to100(ev):
    ev = ev.copy()
    for c, s in [("x", 100 / 120), ("end_x", 100 / 120), ("y", 100 / 80), ("end_y", 100 / 80)]:
        ev[c] = ev[c] * s
    return ev


def match_features(ev, spells, matches):
    keys = ["match_id", "team_id", "player_id"]
    mins = union_minutes(spells)
    base = mins.set_index(keys)
    d = ev[ev.player_id.notna() & ev.period.isin([1, 2])].copy()
    d["player_id"] = d.player_id.astype(int)
    cnt = lambda mask: d[mask].groupby(keys).size()

    # team possession share per match (by passes), used to adjust defensive counts
    passes = ev[ev.type == "Pass"].groupby(["match_id", "team_id"]).size()
    share = passes / passes.groupby(level=0).transform("sum")
    base = base.join(share.rename("team_possession"), on=["match_id", "team_id"])
    padj = (0.5 / (1 - base.team_possession)).clip(0.6, 1.6)

    is_pass = d.type == "Pass"
    open_pass = is_pass & ~d.sub_type.isin(SET_PIECES)
    complete = d.outcome.isna()
    goal_dist = lambda x, y: np.hypot(100 - x, (50 - y) * 0.68 / 1.05)  # metres-ish, 105x68 pitch
    dx = d.end_x - d.x
    prog = (goal_dist(d.end_x, d.end_y) <= 0.75 * goal_dist(d.x, d.y)) & (d.x >= 30)
    base["passes"] = cnt(is_pass)
    base["passes_done"] = cnt(is_pass & complete)
    base["open_passes"] = cnt(open_pass)
    base["long"] = cnt(open_pass & (d.length >= 32))
    base["forward"] = cnt(open_pass & (dx >= 8.3))  # 10 yards on the 120 scale
    base["prog_passes"] = cnt(open_pass & complete & prog)
    base["crosses"] = cnt(open_pass & d.cross.fillna(False).astype(bool))
    base["shot_assists"] = cnt(is_pass & (d.shot_assist.fillna(False).astype(bool) | d.goal_assist.fillna(False).astype(bool)))
    base["receptions"] = cnt(d.type == "Ball Receipt*")
    car = d.type == "Carry"
    base["carries"] = cnt(car)
    base["prog_carries"] = cnt(car & prog & (np.hypot(d.end_x - d.x, d.end_y - d.y) >= 4))
    base["dribbles"] = cnt(d.type == "Dribble")
    base["dribbles_won"] = cnt((d.type == "Dribble") & (d.outcome == "Complete"))
    tackle = (d.type == "Duel") & (d.sub_type == "Tackle")
    base["tackles"] = cnt(tackle)
    base["tackles_won"] = cnt(tackle & d.outcome.isin(["Won", "Success", "Success In Play", "Success Out"]))
    aer_won = d.aerial_won.fillna(False).astype(bool)
    aer_lost = (d.type == "Duel") & (d.sub_type == "Aerial Lost")
    base["aerials_won"] = cnt(aer_won)
    base["aerials"] = cnt(aer_won | aer_lost)
    for k, t in [("pressures", "Pressure"), ("interceptions", "Interception"), ("recoveries", "Ball Recovery"),
                 ("clearances", "Clearance"), ("blocks", "Block"), ("fouls", "Foul Committed"),
                 ("dispossessed", "Dispossessed"), ("miscontrols", "Miscontrol")]:
        base[k] = cnt(d.type == t)
    shot = d.type == "Shot"
    np_shot = shot & (d.sub_type != "Penalty")
    base["shots"] = cnt(np_shot)
    base["npxg"] = d[np_shot].groupby(keys).xg.sum()
    base["goals"] = cnt(shot & (d.outcome == "Goal"))
    base = base.fillna({c: 0 for c in base.columns if c != "team_possession"})
    for k in ["pressures", "tackles", "interceptions", "recoveries", "clearances", "blocks"]:
        base[k + "_adj"] = base[k] * padj.values
    # locations
    on = d[d.type.isin(ONBALL) & d.x.notna()].copy()
    on["final"] = on.x >= 200 / 3
    on["obox"] = (on.x >= 100 - 100 * 16.5 / 105) & ((on.y - 50).abs() <= 100 * 20.16 / 68)
    on["ownbox"] = (on.x <= 100 * 16.5 / 105) & ((on.y - 50).abs() <= 100 * 20.16 / 68)
    on["sx"], on["sy"], on["sy2"], on["sw"] = on.x, on.y, on.y ** 2, (on.y - 50).abs()
    s = on.groupby(keys)[["sx", "sy", "sy2", "sw", "final", "obox", "ownbox"]].sum()
    base = base.join(s).join(on.groupby(keys).size().rename("locations"))
    on["cell"] = np.minimum((on.x / 100 * 6).astype(int), 5) * 5 + np.minimum((on.y / 100 * 5).astype(int), 4)
    grid = pd.crosstab([on[k] for k in keys], on.cell).reindex(columns=range(30), fill_value=0)
    grid.columns = [f"grid_{i}" for i in range(30)]
    base = base.join(grid)
    df_ = d[d.type.isin(DEFENSIVE) & d.x.notna()]
    base = base.join(df_.groupby(keys).x.agg(["sum", "size"]).rename(columns={"sum": "def_sx", "size": "def_n"}))
    num = base.columns.difference(["team_possession"])
    base[num] = base[num].fillna(0)
    out = base.reset_index()
    out["league"] = out.match_id.map(matches.set_index("match_id").league)
    return out


def aggregate(mf, weights=None, ids=None):
    m = mf.copy()
    numeric = m.select_dtypes(include="number").columns.difference(["match_id", "team_id", "player_id"])
    if weights is not None:
        w = m.match_id.map(weights).fillna(0)
        m[numeric] = m[numeric].multiply(w, axis=0)
    m["poss_w"] = m.team_possession * m.minutes
    p = m.groupby("player_id")[list(numeric) + ["poss_w"]].sum()
    if ids is not None:
        p = p.reindex(ids).fillna(0)
    p = p.copy()
    per90 = lambda c: p[c] / p.minutes.replace(0, np.nan) * 90
    for c in ["passes", "receptions", "carries", "dribbles", "prog_passes", "crosses", "shot_assists", "prog_carries",
              "aerials", "fouls", "shots", "dispossessed", "miscontrols", "npxg"]:
        p[c + "90"] = per90(c)
    for c in ["pressures", "tackles", "interceptions", "recoveries", "clearances", "blocks"]:
        p[c + "90"] = per90(c + "_adj")
        p[c + "90_raw"] = per90(c)
    p["long_share"] = p.long / p.open_passes.replace(0, np.nan) * 100
    p["forward_share"] = p.forward / p.open_passes.replace(0, np.nan) * 100
    loc = p.locations.replace(0, np.nan)
    p["avg_x"] = p.sx / loc
    p["avg_y"] = p.sy / loc
    p["spread_y"] = np.sqrt((p.sy2 / loc - p.avg_y ** 2).clip(lower=0))
    p["width"] = p.sw / loc
    p["final_third"] = p.final / loc * 100
    p["opp_box"] = p.obox / loc * 100
    p["own_box"] = p.ownbox / loc * 100
    p["def_height"] = p.def_sx / p.def_n.replace(0, np.nan)
    p["npxg_per_shot"] = p.npxg / p.shots.replace(0, np.nan)
    # context: smoothed rates (unknown stays NaN in the *_raw columns)
    for out, col, won, k in [("pass_pct", "passes", "passes_done", 30), ("dribble_pct", "dribbles", "dribbles_won", 10),
                             ("aerial_pct", "aerials", "aerials_won", 20), ("tackle_pct", "tackles", "tackles_won", 20)]:
        prior = p[won].sum() / max(1, p[col].sum())
        p[out] = (p[won] + k * prior) / (p[col] + k) * 100
        p[out + "_raw"] = p[won] / p[col].replace(0, np.nan) * 100
    p["team_possession"] = p.poss_w / p.minutes.replace(0, np.nan) * 100
    for i in range(30):
        p[f"grid_{i}"] = p[f"grid_{i}"] / loc
    return p


def representation(p, cols=None, spatial=False, var=0.85):
    cols = cols or list(FEATURES)
    imp = SimpleImputer(strategy="median"); x = imp.fit_transform(p[cols])
    lo, hi = np.quantile(x, [0.01, 0.99], axis=0); x = np.clip(x, lo, hi)
    sc = StandardScaler(); z = sc.fit_transform(x)
    if spatial:
        gz = StandardScaler().fit_transform(np.sqrt(p[[f"grid_{i}" for i in range(30)]].fillna(0).values))
        z = np.column_stack([z / np.sqrt(z.shape[1]), gz / np.sqrt(30)])
    pc = PCA(n_components=var, svd_solver="full"); space = pc.fit_transform(z)
    return space, pc, dict(columns=cols, median=imp.statistics_, low=lo, high=hi, mean=sc.mean_, scale=sc.scale_)


def aligned(ref, other, k):
    t = np.zeros((k, k), int)
    for a, b in zip(ref, other):
        t[a, b] += 1
    r, c = linear_sum_assignment(-t); mp = dict(zip(c, r))
    return np.array([mp.get(v, v) for v in other])


def choose_k(space, kmin=3, kmax=5, min_size=15, min_seed_ari=0.8):
    rows, sols = [], {}
    for k in range(kmin, kmax + 1):
        m = KMeans(k, random_state=17, n_init=30).fit(space); sols[k] = m
        seed = np.mean([adjusted_rand_score(m.labels_, KMeans(k, random_state=s, n_init=10).fit_predict(space)) for s in [2, 7, 31, 53, 97]])
        sizes = np.bincount(m.labels_, minlength=k)
        rows.append(dict(k=k, silhouette=silhouette_score(space, m.labels_), seed_ari=seed, smallest=int(sizes.min()), inertia=m.inertia_))
    c = pd.DataFrame(rows)
    # Styles form a continuum, so silhouette always prefers the coarsest split (k=2). For archetypes we want
    # the most detailed grouping that is still reproducible: the largest k whose seed agreement >= min_seed_ari
    # and whose smallest group has >= min_size players. Match-bootstrap ARI then tests it against new samples.
    c["eligible"] = (c.seed_ari >= min_seed_ari) & (c.smallest >= min_size)
    k = int(c[c.eligible].k.max()) if c.eligible.any() else kmin
    return k, sols, c


def describe(pct_center, n=3):
    """Plain description of a sub-role from its within-group percentile centroid."""
    hi = pct_center.sort_values(ascending=False).head(n); lo = pct_center.sort_values().head(2)
    lab = lambda k: FEATURES[k][0].lower()
    return ("More " + ", ".join(lab(k) for k in hi.index) + "; less " + ", ".join(lab(k) for k in lo.index) +
            " than others in the group."), list(hi.index), list(lo.index)


# Archetype names, chosen after inspecting each sub-role's high/low features and example players
# (outputs/profiling_sb/subroles.csv). Matched by the defining features rather than cluster ids, and assigned
# one-to-one within a group, so a rerun keeps sensible, unique names.
NAMES = {
    "Centre-back": [
        ("Ball-playing centre-back", {"carries90", "passes90", "receptions90", "prog_passes90", "prog_carries90"}, {"own_box", "long_share"}),
        ("Box-defending stopper", {"own_box", "long_share", "clearances90", "aerials90", "blocks90"}, {"passes90", "carries90"}),
        ("Wide centre-back (back three)", {"width", "avg_x", "crosses90", "spread_y", "def_height"}, {"own_box", "clearances90"}),
    ],
    "Full-back": [
        ("Touchline progressor", {"width", "prog_passes90", "forward_share"}, {"opp_box", "spread_y"}),
        ("Overlapping attacking full-back", {"avg_x", "final_third", "opp_box", "crosses90", "dribbles90"}, {"clearances90", "own_box"}),
        ("Possession full-back", {"carries90", "receptions90", "passes90"}, {"own_box", "forward_share", "long_share"}),
        ("Conservative full-back", {"own_box", "long_share", "clearances90"}, {"carries90", "receptions90", "passes90"}),
        ("Aggressive defending full-back", {"pressures90", "tackles90", "aerials90", "fouls90"}, {"crosses90", "shot_assists90"}),
    ],
    "Midfielder": [
        ("Deep controller", {"carries90", "passes90", "recoveries90", "receptions90"}, {"opp_box", "width", "avg_x"}),
        ("Long-passing creator", {"shot_assists90", "crosses90", "long_share"}, {"tackles90", "blocks90"}),
        ("Holding screen", {"own_box", "clearances90", "blocks90", "aerials90"}, {"avg_x", "def_height"}),
        ("Progressive box-to-box", {"receptions90", "prog_carries90", "def_height", "dribbles90"}, {"own_box", "clearances90"}),
        ("Box-crashing midfielder", {"opp_box", "final_third", "avg_x", "shots90"}, {"passes90", "carries90"}),
    ],
    "Attacking mid / winger": [
        ("Central creative playmaker", {"passes90", "carries90", "receptions90", "shot_assists90"}, {"opp_box", "width"}),
        ("Hard-working wide midfielder", {"own_box", "clearances90", "pressures90", "tackles90", "def_height"}, {"avg_x"}),
        ("Inside forward / goal threat", {"opp_box", "final_third", "npxg_per_shot", "shots90"}, {"prog_passes90", "passes90"}),
        ("Dribbling winger", {"prog_carries90", "dribbles90", "crosses90", "width"}, {"own_box", "clearances90"}),
    ],
    "Forward": [
        ("Dribbling channel-runner", {"carries90", "prog_carries90", "dribbles90"}, {"opp_box", "aerials90"}),
        ("Wide-roaming worker", {"spread_y", "tackles90", "width", "pressures90"}, {"avg_x", "def_height"}),
        ("Penalty-box target striker", {"opp_box", "aerials90", "fouls90"}, {"width", "spread_y"}),
        ("Mobile front-line finisher", {"final_third", "avg_x", "def_height", "shots90"}, {"aerials90", "fouls90"}),
    ],
}


def name_subroles(group, profiles):
    """profiles: list of (hi, lo) feature lists per sub-role. Returns one unique name per sub-role."""
    cands = NAMES.get(group, [])
    if not cands:
        return [f"{group}: " + " + ".join(FEATURES[f][0].lower() for f in hi[:2]) for hi, _ in profiles]
    score = np.zeros((len(profiles), len(cands)))
    for i, (hi, lo) in enumerate(profiles):
        for j, (_, hset, lset) in enumerate(cands):
            score[i, j] = sum((3 - r) for r, f in enumerate(hi[:3]) if f in hset) + sum((2 - r) for r, f in enumerate(lo[:2]) if f in lset)
    names = [None] * len(profiles)
    r, c = linear_sum_assignment(-score)
    for i, j in zip(r, c):
        if score[i, j] > 0:
            names[i] = cands[j][0]
    return [n or f"{group}: " + " + ".join(FEATURES[f][0].lower() for f in profiles[i][0][:2]) for i, n in enumerate(names)]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-minutes", type=float, default=900)
    ap.add_argument("--bootstraps", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    matches = pd.read_parquet(DATA / "matches.parquet")
    cache = WORK / "match_features.pkl"
    if cache.exists():
        mf = pd.read_pickle(cache); spells = pd.read_pickle(WORK / "spells.pkl"); audit = pd.read_pickle(WORK / "audit.pkl")
    else:
        cols = ["match_id", "period", "minute", "second", "type", "team_id", "player_id", "x", "y", "end_x", "end_y", "outcome",
                "sub_type", "length", "cross", "shot_assist", "goal_assist", "aerial_won", "xg"]
        ev = to100(read_events(cols))
        lineups = read_lineups()
        spells, audit = reconstruct_minutes(lineups, period_lengths(ev))
        print(f"Spells {len(spells)}; flagged team-matches {audit.roster_issue.sum()} of {len(audit)}", flush=True)
        mf = match_features(ev, spells, matches)
        mf.to_pickle(cache); spells.to_pickle(WORK / "spells.pkl"); audit.to_pickle(WORK / "audit.pkl")
        del ev
    audit.to_csv(OUT / "minutes_audit.csv", index=False)

    # player-season table
    allp = aggregate(mf)
    meta = spells.groupby("player_id").agg(name=("player", "first"), nickname=("nickname", "first"))
    gm = spells.groupby(["player_id", "group"]).minutes.sum().unstack(fill_value=0)
    allp["group"] = gm.idxmax(axis=1)
    allp["group_share"] = (gm.max(axis=1) / gm.sum(axis=1).replace(0, np.nan) * 100)
    pos = spells.groupby(["player_id", "position"]).minutes.sum().reset_index().sort_values("minutes", ascending=False).drop_duplicates("player_id").set_index("player_id").position
    allp["main_position"] = pos
    tm = spells.groupby(["player_id", "team_id", "team"]).minutes.sum().reset_index().sort_values("minutes", ascending=False)
    lg = mf.groupby(["player_id", "league"]).minutes.sum().reset_index().sort_values("minutes", ascending=False).drop_duplicates("player_id").set_index("player_id").league
    allp["league"] = lg
    allp["display"] = meta.nickname.fillna(meta.name).reindex(allp.index)
    allp["sent_off"] = spells.groupby("player_id").sent_off.sum()
    flagged = set(map(tuple, audit.loc[audit.roster_issue, ["match_id", "team_id"]].values))
    mf["flag"] = [(a, b) in flagged for a, b in zip(mf.match_id, mf.team_id)]
    allp["flagged_matches"] = mf.groupby("player_id").flag.sum()
    p = allp[(allp.minutes >= args.min_minutes) & allp.group.isin(GROUP_ORDER)].copy()
    print(f"Eligible outfield players: {len(p)} (of {len(allp)})", flush=True)

    # ---- global style space (for similarity and the map)
    space, pc, transform = representation(p)
    spatial_space, spatial_pc, _ = representation(p, spatial=True)
    D = cdist(space, space); np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    top10, top5 = order[:, :25], order[:, :5]  # keep 25 so filtered views (league, team) still have stability info
    print(f"PCA dims {space.shape[1]} ({pc.explained_variance_ratio_.sum():.0%}); map {pc.explained_variance_ratio_[:2].sum():.0%}", flush=True)

    # ---- level 2: sub-roles within each position group (own scaling + PCA per group)
    pct_global = p[list(FEATURES)].rank(pct=True) * 100
    p["subrole_id"] = -1
    subroles, cand_all, group_spaces, group_models = [], [], {}, {}
    for gname in GROUP_ORDER:
        idx = np.flatnonzero(p.group.values == gname)
        gp = p.iloc[idx]
        gs, gpc, _ = representation(gp)
        k, sols, cand = choose_k(gs)
        cand.insert(0, "group", gname); cand_all.append(cand)
        lab = sols[k].labels_
        group_spaces[gname] = (idx, gs); group_models[gname] = (k, lab)
        gpct = gp[list(FEATURES)].rank(pct=True) * 100
        for c in range(k):
            mem = np.flatnonzero(lab == c)
            center = gpct.iloc[mem].mean()
            desc, hi, lo = describe(center)
            reps = mem[np.argsort(np.linalg.norm(gs[mem] - sols[k].cluster_centers_[c], axis=1))[:5]]
            subroles.append(dict(group=gname, local_id=c, name=None, description=desc, count=len(mem),
                                 top=hi, low=lo, percentiles=center.to_dict(),
                                 examples=[str(x) for x in gp.display.iloc[reps].values]))
            p.iloc[idx[mem], p.columns.get_loc("subrole_id")] = len(subroles) - 1
        mine = [s_ for s_ in subroles if s_["group"] == gname]
        for s_, nm in zip(mine, name_subroles(gname, [(s_["top"], s_["low"]) for s_ in mine])):
            s_["name"] = nm
    # de-duplicate names within a group
    seen = {}
    for s in subroles:
        key = (s["group"], s["name"]); seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            s["name"] += f" ({seen[key]})"
    for i, s in enumerate(subroles):
        s["id"] = i
    p["subrole"] = [subroles[i]["name"] for i in p.subrole_id]
    pd.concat(cand_all).to_csv(OUT / "subrole_candidates.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "percentiles"} for s in subroles]).to_csv(OUT / "subroles.csv", index=False)
    for s in subroles:
        print(f"  {s['group']:<24} {s['name']:<34} n={s['count']:<4} e.g. {', '.join(s['examples'][:3])}", flush=True)

    # ---- bootstrap: whole-match resampling, rebuild rates/transforms on the fixed cohort
    rng = np.random.default_rng(2026); mids = mf.match_id.unique()
    freq = np.zeros((len(p), len(p)), dtype=np.float32)
    stab_rows = []; sub_votes = np.zeros(len(p))
    for b in range(args.bootstraps):
        w = pd.Series(rng.choice(mids, len(mids), replace=True)).value_counts()
        bp = aggregate(mf, w, p.index)
        bs, _, _ = representation(bp)
        bd = cdist(bs, bs); np.fill_diagonal(bd, np.inf); bn = np.argsort(bd, axis=1)[:, :5]
        freq[np.repeat(np.arange(len(p)), 5), bn.ravel()] += 1
        overlap = np.mean([len(set(a) & set(c)) / 5 for a, c in zip(top5, bn)])
        row = dict(replicate=b, top5_overlap=overlap)
        for gname in GROUP_ORDER:
            idx, _ = group_spaces[gname]; k, lab = group_models[gname]
            gb, _, _ = representation(bp.iloc[idx])
            bl = KMeans(k, random_state=b, n_init=10).fit_predict(gb)
            row["ari_" + gname] = adjusted_rand_score(lab, bl)
            sub_votes[idx] += (aligned(lab, bl, k) == lab)
        stab_rows.append(row)
        if (b + 1) % 5 == 0:
            print(f"bootstrap {b + 1}/{args.bootstraps}", flush=True)
    B = max(1, args.bootstraps)
    freq /= B
    stab = pd.DataFrame(stab_rows); stab.to_csv(OUT / "bootstrap_stability.csv", index=False)

    # ---- league / team context checks: how often top-5 neighbours share league or team vs chance
    lg_arr = np.asarray(p.league.values, dtype=object); tm_main = np.asarray(tm.drop_duplicates("player_id").set_index("player_id").team.reindex(p.index).values, dtype=object)
    same_league = np.mean(lg_arr[top5] == lg_arr[:, None]); base_league = np.mean([(lg_arr == l).mean() for l in lg_arr])
    same_team = np.mean(tm_main[top5] == tm_main[:, None]); base_team = np.mean([((tm_main == t).sum() - 1) / (len(p) - 1) for t in tm_main])
    g_arr = np.asarray(p.group.values, dtype=object); same_group = np.mean(g_arr[top5] == g_arr[:, None])

    loadings = pd.DataFrame(pc.components_.T, index=list(FEATURES), columns=[f"PC{i + 1}" for i in range(space.shape[1])])
    loadings.to_csv(OUT / "pca_loadings.csv")
    corr = p[list(FEATURES)].corr(); corr.to_csv(OUT / "feature_correlations.csv")
    pd.crosstab(p.group, p.subrole).to_csv(OUT / "group_subrole_counts.csv")

    # ---- records
    teams_of = tm.groupby("player_id").team.apply(list)
    records = []
    for i, (pid, r) in enumerate(p.iterrows()):
        nb = top10[i]
        records.append(dict(
            id=int(pid), name=r.display, full_name=meta.name.get(pid), league=r.league, teams=teams_of.get(pid, []),
            group=r.group, group_share=r.group_share, position=r.main_position, minutes=r.minutes,
            subrole=r.subrole, subrole_id=int(r.subrole_id), subrole_stability=float(sub_votes[i] / B),
            map=space[i, :2], space=space[i], spatial_map=spatial_space[i, :2],
            features={c: r[c] for c in FEATURES}, percentiles=pct_global.iloc[i].to_dict(),
            context={c: r[c] for c in CONTEXT if c in r}, raw={c: r.get(c) for c in ["pass_pct_raw", "dribble_pct_raw", "aerial_pct_raw", "tackle_pct_raw", "passes", "dribbles", "aerials", "tackles", "npxg", "shots"]},
            heatmap=[r[f"grid_{j}"] for j in range(30)],
            neighbours=[dict(id=int(p.index[j]), distance=float(D[i, j]), stability=float(freq[i, j])) for j in nb],
            neighbour_stability=float(np.mean([freq[i, j] for j in top5[i]])),
            confidence="Check roster" if r.flagged_matches > 0 else "Lineup minutes",
        ))
    summary = dict(
        source="StatsBomb Open Data 2015/16: Premier League, La Liga, Serie A, Ligue 1",
        matches=int(mf.match_id.nunique()), eligible=len(p), all_players=len(allp), min_minutes=args.min_minutes,
        group_counts=p.group.value_counts().reindex(GROUP_ORDER).to_dict(), subroles=len(subroles),
        pca_components=space.shape[1], pca_variance=pc.explained_variance_ratio_.sum(), map_variance=pc.explained_variance_ratio_[:2].sum(),
        bootstraps=args.bootstraps, top5_overlap=float(stab.top5_overlap.mean()),
        subrole_ari={g: float(stab["ari_" + g].mean()) for g in GROUP_ORDER} if len(stab) else {},
        stable_neighbour_share=float((freq[np.arange(len(p))[:, None], top5] >= 0.6).mean()),
        top5_same_league=float(same_league), chance_same_league=float(base_league),
        top5_same_team=float(same_team), chance_same_team=float(base_team), top5_same_group=float(same_group),
        roster_issues=int(audit.roster_issue.sum()), team_matches=len(audit))
    payload = dict(summary=summary, players=records, subroles=subroles, groups=GROUP_ORDER,
                   features=[dict(key=k, label=v[0], group=v[1]) for k, v in FEATURES.items()],
                   context=[dict(key=k, label=v) for k, v in CONTEXT.items()], radar=RADAR,
                   pca_variance=pc.explained_variance_ratio_, loadings=loadings.reset_index(names="feature").to_dict("records"),
                   generated=datetime.now(timezone.utc).isoformat())
    text = json.dumps(clean_json(payload), ensure_ascii=False, allow_nan=False)
    (OUT / "profiles.json").write_text(text, encoding="utf-8")
    site = ROOT / "docs"
    if site.exists():  # keep the GitHub Pages explorer in sync
        (site / "profiles.json").write_text(text, encoding="utf-8")
    exp = p.drop(columns=[f"grid_{j}" for j in range(30)]).copy()
    exp.to_csv(OUT / "player_profiles.csv")
    (OUT / "model_transform.json").write_text(json.dumps(clean_json(dict(transform=transform, pca_components=pc.components_, pca_mean=pc.mean_))), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
