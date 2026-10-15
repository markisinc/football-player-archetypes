"""Download StatsBomb open data (2015/16: Premier League, La Liga, Serie A, Ligue 1) and flatten
to compact parquet files: events_partNNN, lineups_partNNN (position spells), matches. Resumable.

Run: python src/sb_load.py   (writes data/statsbomb/*.parquet; raw JSON is not kept)
Data: StatsBomb Open Data, https://github.com/statsbomb/open-data (attribution required).
"""
from pathlib import Path
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "statsbomb"
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SEASON = 27  # 2015/2016
COMPETITIONS = {2: "Premier League", 11: "La Liga", 12: "Serie A", 7: "Ligue 1"}  # Bundesliga 2015/16 is one team only


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 - retry any transient failure
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def g(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def name(d, *path):
    return g(d, *path, "name")


def flatten_events(match_id, events):
    rows = []
    for e in events:
        t = name(e, "type")
        loc = e.get("location") or [None, None]
        r = dict(match_id=match_id, id=e["id"], index=e["index"], period=e["period"], minute=e["minute"], second=e["second"],
                 timestamp=e.get("timestamp"), type=t, team_id=g(e, "team", "id"), team=name(e, "team"),
                 player_id=g(e, "player", "id"), player=name(e, "player"), position=name(e, "position"),
                 x=loc[0], y=loc[1], possession=e.get("possession"), possession_team_id=g(e, "possession_team", "id"),
                 play_pattern=name(e, "play_pattern"), under_pressure=bool(e.get("under_pressure", False)),
                 counterpress=bool(e.get("counterpress", False)), duration=e.get("duration"))
        end = None
        if t == "Pass":
            p = e["pass"]; end = p.get("end_location")
            r.update(outcome=name(p, "outcome"), sub_type=name(p, "type"), height=name(p, "height"), body_part=name(p, "body_part"),
                     length=p.get("length"), cross=bool(p.get("cross", False)), switch=bool(p.get("switch", False)),
                     through_ball=bool(p.get("through_ball", False)), cut_back=bool(p.get("cut_back", False)),
                     shot_assist=bool(p.get("shot_assist", False)), goal_assist=bool(p.get("goal_assist", False)),
                     aerial_won=bool(p.get("aerial_won", False)))
        elif t == "Carry":
            end = g(e, "carry", "end_location")
        elif t == "Shot":
            s = e["shot"]; end = s.get("end_location")
            r.update(outcome=name(s, "outcome"), sub_type=name(s, "type"), body_part=name(s, "body_part"),
                     xg=s.get("statsbomb_xg"), aerial_won=bool(s.get("aerial_won", False)))
        elif t == "Dribble":
            r.update(outcome=name(e.get("dribble", {}), "outcome"))
        elif t == "Duel":
            r.update(outcome=name(e.get("duel", {}), "outcome"), sub_type=name(e.get("duel", {}), "type"))
        elif t == "Interception":
            r.update(outcome=name(e.get("interception", {}), "outcome"))
        elif t == "Clearance":
            r.update(aerial_won=bool(g(e, "clearance", "aerial_won")), body_part=name(e.get("clearance", {}), "body_part"))
        elif t == "Miscontrol":
            r.update(aerial_won=bool(g(e, "miscontrol", "aerial_won")))
        elif t == "Ball Recovery":
            r.update(outcome="Failure" if g(e, "ball_recovery", "recovery_failure") else None)
        elif t == "Foul Committed":
            r.update(card=name(e.get("foul_committed", {}), "card"))
        elif t == "Bad Behaviour":
            r.update(card=name(e.get("bad_behaviour", {}), "card"))
        elif t == "Goal Keeper":
            r.update(sub_type=name(e.get("goalkeeper", {}), "type"), outcome=name(e.get("goalkeeper", {}), "outcome"))
        elif t == "Substitution":
            r.update(outcome=name(e.get("substitution", {}), "outcome"), recipient_id=g(e, "substitution", "replacement", "id"))
        if end:
            r.update(end_x=end[0], end_y=end[1])
        rows.append(r)
    return rows


def flatten_lineups(match_id, lineups):
    rows = []
    for team in lineups:
        for p in team["lineup"]:
            for s in p.get("positions", []):
                rows.append(dict(match_id=match_id, team_id=team["team_id"], team=team["team_name"], player_id=p["player_id"],
                                 player=p["player_name"], nickname=p.get("player_nickname"), position=s.get("position"),
                                 from_clock=s.get("from"), to_clock=s.get("to"), from_period=s.get("from_period"),
                                 to_period=s.get("to_period"), start_reason=s.get("start_reason"), end_reason=s.get("end_reason")))
    return rows


def fetch(match_id):
    ev = get(f"{BASE}/events/{match_id}.json")
    lu = get(f"{BASE}/lineups/{match_id}.json")
    return flatten_events(match_id, ev), flatten_lineups(match_id, lu)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matches = []
    for cid, cname in COMPETITIONS.items():
        for m in get(f"{BASE}/matches/{cid}/{SEASON}.json"):
            matches.append(dict(match_id=m["match_id"], league=cname, date=m["match_date"], home_team_id=m["home_team"]["home_team_id"],
                                home=m["home_team"]["home_team_name"], away_team_id=m["away_team"]["away_team_id"],
                                away=m["away_team"]["away_team_name"], home_score=m["home_score"], away_score=m["away_score"]))
    matches = pd.DataFrame(matches)
    matches.to_parquet(OUT / "matches.parquet", index=False)
    done = set()
    parts = sorted(OUT.glob("events_part*.parquet"))
    for p in parts:
        done |= set(pd.read_parquet(p, columns=["match_id"]).match_id.unique())
    todo = [m for m in matches.match_id if m not in done]
    print(f"{len(matches)} matches, {len(todo)} to download", flush=True)
    EV_COLS = ["match_id", "id", "index", "period", "minute", "second", "timestamp", "type", "team_id", "team", "player_id",
               "player", "position", "x", "y", "end_x", "end_y", "possession", "possession_team_id", "play_pattern",
               "under_pressure", "counterpress", "duration", "outcome", "sub_type", "height", "body_part", "length", "cross",
               "switch", "through_ball", "cut_back", "shot_assist", "goal_assist", "aerial_won", "xg", "card", "recipient_id"]
    n = len(parts)
    for start in range(0, len(todo), 100):  # bounded batches keep memory flat
        ev_buf, lu_buf = [], []
        with ThreadPoolExecutor(8) as pool:
            for ev, lu in pool.map(fetch, todo[start:start + 100]):
                ev_buf += ev; lu_buf += lu
        ev_df = pd.DataFrame(ev_buf).reindex(columns=EV_COLS)
        for c in ["cross", "switch", "through_ball", "cut_back", "shot_assist", "goal_assist", "aerial_won"]:
            ev_df[c] = ev_df[c].astype("boolean")
        ev_df.to_parquet(OUT / f"events_part{n:03d}.parquet", index=False)
        pd.DataFrame(lu_buf).to_parquet(OUT / f"lineups_part{n:03d}.parquet", index=False)
        n += 1
        print(f"{min(start + 100, len(todo))}/{len(todo)}", flush=True)
    print("done:", len(list(OUT.glob("events_part*.parquet"))), "event files", flush=True)


def read_events(columns=None):
    """Read all event parts (optionally a column subset) into one DataFrame."""
    parts = [pd.read_parquet(p, columns=columns) for p in sorted(OUT.glob("events_part*.parquet"))]
    ev = pd.concat(parts, ignore_index=True)
    for c in ["type", "outcome", "sub_type"]:
        if c in ev:
            ev[c] = ev[c].astype("category")
    return ev


def read_lineups():
    return pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("lineups_part*.parquet"))], ignore_index=True)


if __name__ == "__main__":
    sys.exit(main())
