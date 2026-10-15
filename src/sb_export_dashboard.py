"""Flatten outputs/profiling_sb/profiles.json into tidy CSVs for dashboards (values rounded as displayed).

Run: python src/sb_export_dashboard.py   -> outputs/profiling_sb/dashboard/*.csv
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "profiling_sb" / "profiles.json"
OUT = ROOT / "outputs" / "profiling_sb" / "dashboard"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(SRC.read_text(encoding="utf-8"))
    P = d["players"]; feats = [f["key"] for f in d["features"]]
    byid = {p["id"]: p for p in P}
    space = np.array([p["space"] for p in P])
    rng = np.random.default_rng(7)
    a, b = rng.integers(0, len(P), 40000), rng.integers(0, len(P), 40000)
    keep = a != b
    typical = float(np.median(np.linalg.norm(space[a[keep]] - space[b[keep]], axis=1)))

    rows = []
    for p in P:
        r = dict(id=p["id"], name=p["name"], full_name=p["full_name"], league=p["league"], team=p["teams"][0],
                 teams=" / ".join(p["teams"]), group=p["group"], position=p["position"], minutes=round(p["minutes"]),
                 archetype=p["subrole"], archetype_id=p["subrole_id"], archetype_stability=round(100 * p["subrole_stability"]),
                 map_x=round(p["map"][0], 3), map_y=round(p["map"][1], 3),
                 team_possession=None if p["context"].get("team_possession") is None else round(p["context"]["team_possession"]),
                 goals=round(p["context"].get("goals") or 0), npxg90=round(p["context"].get("npxg90") or 0, 2),
                 pass_pct=None if p["raw"].get("pass_pct_raw") is None else round(p["raw"]["pass_pct_raw"]),
                 lineup_flag=p["confidence"] != "Lineup minutes")
        for k in feats:
            v = p["percentiles"].get(k)
            r["p_" + k] = None if v is None else round(v)  # e.g. chance quality for a player with no shots
        for i, v in enumerate(p["heatmap"]):
            r[f"h{i}"] = round(v or 0, 4)
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "sb_players.csv", index=False)

    nb = []
    for p in P:
        for rank, n in enumerate(p["neighbours"], 1):
            o = byid[n["id"]]
            nb.append(dict(player_id=p["id"], rank=rank, neighbour_id=o["id"], neighbour=o["name"], neighbour_team=o["teams"][0],
                           neighbour_teams=" / ".join(o["teams"]), neighbour_league=o["league"], neighbour_group=o["group"],
                           neighbour_archetype=o["subrole"], match_pct=max(0, round(100 * (1 - n["distance"] / typical))),
                           holds_up_pct=round(100 * n["stability"])))
    pd.DataFrame(nb).to_csv(OUT / "sb_neighbours.csv", index=False)

    arch, prof = [], []
    flab = {f["key"]: f["label"] for f in d["features"]}
    for s in d["subroles"]:
        mem = [p for p in P if p["subrole_id"] == s["id"]]
        arch.append(dict(id=s["id"], group=s["group"], name=s["name"], description=s["description"], players=s["count"],
                         core_players=sum(p["subrole_stability"] >= 0.8 for p in mem), examples=", ".join(s["examples"][:4]),
                         top_features=",".join(s["top"]), low_features=",".join(s["low"])))
        for k in feats:
            prof.append(dict(archetype_id=s["id"], feature=k, label=flab[k], percentile_in_group=None if s["percentiles"].get(k) is None else round(s["percentiles"][k])))
    pd.DataFrame(arch).to_csv(OUT / "sb_archetypes.csv", index=False)
    pd.DataFrame(prof).to_csv(OUT / "sb_archetype_profiles.csv", index=False)
    pd.DataFrame([dict(key=f["key"], label=f["label"], category=f["group"]) for f in d["features"]]).to_csv(OUT / "sb_features.csv", index=False)

    s = d["summary"]
    summ = [("matches", s["matches"]), ("eligible_players", s["eligible"]), ("min_minutes", int(s["min_minutes"])),
            ("archetypes", s["subroles"]), ("pca_dimensions", s["pca_components"]), ("pca_variance_pct", round(100 * s["pca_variance"])),
            ("map_variance_pct", round(100 * s["map_variance"])), ("bootstraps", s["bootstraps"]),
            ("top5_overlap_pct", round(100 * s["top5_overlap"])), ("stable_link_pct", round(100 * s["stable_neighbour_share"])),
            ("same_league_pct", round(100 * s["top5_same_league"])), ("same_league_chance_pct", round(100 * s["chance_same_league"])),
            ("core_player_pct", round(100 * np.mean([p["subrole_stability"] >= 0.8 for p in P]))),
            ("typical_distance", round(typical, 3)), ("flagged_lineups", s["roster_issues"]), ("team_matches", s["team_matches"])]
    summ += [(f"ari_{g}", round(v, 2)) for g, v in s["subrole_ari"].items()]
    pd.DataFrame(summ, columns=["metric", "value"]).to_csv(OUT / "sb_summary.csv", index=False)
    for f in sorted(OUT.glob("*.csv")):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
