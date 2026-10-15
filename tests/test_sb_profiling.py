"""Invariants for the StatsBomb profiling pipeline: lineup minutes and exported outputs."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sb_profiling import reconstruct_minutes, union_minutes, aggregate, OUT  # noqa: E402

PROFILES = next((f for f in [OUT / "profiles.json", ROOT / "docs" / "profiles.json"] if f.exists()), OUT / "profiles.json")


def lineup(rows):
    cols = ["match_id", "team_id", "team", "player_id", "player", "nickname", "position", "from_clock", "to_clock",
            "from_period", "to_period", "start_reason", "end_reason"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


# first half 47:00 long, second half 49:00 long (clock 45:00 -> 94:00)
LENGTHS = pd.DataFrame({1: [47 * 60], 2: [49 * 60]}, index=pd.Index([1], name="match_id"))
TOTAL = 96.0


def xi(extra):
    """Ten fixed starters playing the whole match plus the given rows."""
    base = [(1, 1, "A", 100 + i, f"P{i}", None, "Center Midfield", "00:00", None, 1, None, "Starting XI", "Final Whistle")
            for i in range(10)]
    return lineup(base + extra)


class Minutes(unittest.TestCase):
    def test_substitution_conserves_minutes_and_handles_stoppage(self):
        lu = xi([(1, 1, "A", 1, "Starter", None, "Center Forward", "00:00", "70:00", 1, 2, "Starting XI", "Substitution - Off (Tactical)"),
                 (1, 1, "A", 2, "Sub", None, "Center Forward", "70:00", None, 2, None, "Substitution - On (Tactical)", "Final Whistle")])
        sp, audit = reconstruct_minutes(lu, LENGTHS)
        m = union_minutes(sp).set_index("player_id").minutes
        self.assertAlmostEqual(m[1], 47 + 25)          # full first half incl. stoppage + 25 of the second
        self.assertAlmostEqual(m[2], TOTAL - 72)
        self.assertAlmostEqual(m[1] + m[2], TOTAL)       # the slot is covered exactly once
        self.assertFalse(audit.roster_issue.any())

    def test_halftime_sub_starts_after_first_half_stoppage(self):
        lu = xi([(1, 1, "A", 1, "Off", None, "Left Back", "00:00", "45:00", 1, 2, "Starting XI", "Substitution - Off (Tactical)"),
                 (1, 1, "A", 2, "On", None, "Left Back", "45:00", None, 2, None, "Substitution - On (Tactical)", "Final Whistle")])
        sp, _ = reconstruct_minutes(lu, LENGTHS)
        m = union_minutes(sp).set_index("player_id").minutes
        self.assertAlmostEqual(m[1], 47)
        self.assertAlmostEqual(m[2], 49)

    def test_ghost_spell_after_substitution_is_removed(self):
        lu = xi([(1, 1, "A", 1, "Off", None, "Left Wing", "00:00", "45:00", 1, 2, "Starting XI", "Substitution - Off (Tactical)"),
                 (1, 1, "A", 1, "Off", None, "Left Wing", "45:00", None, 2, None, "Tactical Shift", "Final Whistle"),
                 (1, 1, "A", 2, "On", None, "Left Wing", "45:00", None, 2, None, "Substitution - On (Tactical)", "Final Whistle")])
        sp, audit = reconstruct_minutes(lu, LENGTHS)
        m = union_minutes(sp).set_index("player_id").minutes
        self.assertAlmostEqual(m[1], 47)
        self.assertFalse(audit.roster_issue.any())

    def test_mislabelled_starter_counts_as_starter(self):
        lu = xi([(1, 1, "A", 1, "X", None, "Goalkeeper", "00:00", None, 1, None, "Tactical Shift", "Final Whistle")])
        sp, audit = reconstruct_minutes(lu, LENGTHS)
        self.assertEqual(audit.starters.iloc[0], 11)

    def test_sent_off_player_is_flagged_and_team_drops_to_ten(self):
        lu = xi([(1, 1, "A", 1, "Red", None, "Center Back", "00:00", "60:00", 1, 2, "Starting XI", "Sent Off")])
        sp, audit = reconstruct_minutes(lu, LENGTHS)
        self.assertTrue(sp[sp.player_id == 1].sent_off.all())
        self.assertEqual(audit.min_active.iloc[0], 10)
        self.assertFalse(audit.roster_issue.any())

    def test_overlapping_spells_do_not_double_count(self):
        sp = pd.DataFrame(dict(match_id=1, team_id=1, player_id=1, start=[0, 30, 40], end=[35, 50, 45]))
        self.assertAlmostEqual(union_minutes(sp).minutes.iloc[0], 50)


class Rates(unittest.TestCase):
    def test_per90_and_unknown_rates(self):
        mf = pd.DataFrame(dict(match_id=[1, 2], team_id=1, player_id=7, minutes=[45.0, 45.0], team_possession=[0.5, 0.5],
                               passes=[20, 20], passes_done=[18, 18], dribbles=[0, 0], dribbles_won=[0, 0]))
        for c in ["receptions", "carries", "prog_passes", "crosses", "shot_assists", "prog_carries", "aerials", "aerials_won",
                  "fouls", "shots", "dispossessed", "miscontrols", "npxg", "long", "forward", "open_passes", "tackles", "tackles_won",
                  "sx", "sy", "sy2", "sw", "final", "obox", "ownbox", "locations", "def_sx", "def_n"] + \
                 [f"{k}{s}" for k in ["pressures", "tackles", "interceptions", "recoveries", "clearances", "blocks"] for s in ["", "_adj"]] + \
                 [f"grid_{i}" for i in range(30)]:
            if c not in mf:
                mf[c] = 0.0
        p = aggregate(mf)
        self.assertAlmostEqual(p.loc[7, "passes90"], 40)
        self.assertTrue(np.isnan(p.loc[7, "dribble_pct_raw"]))   # no attempts = unknown, not 0%


@unittest.skipUnless(PROFILES.exists(), "run src/sb_profiling.py first")
class Outputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.loads(PROFILES.read_text(encoding="utf-8"))

    def test_neighbours_distinct_sorted_and_exclude_self(self):
        ids = {p["id"] for p in self.d["players"]}
        for p in self.d["players"]:
            nb = [n["id"] for n in p["neighbours"]]
            self.assertNotIn(p["id"], nb)
            self.assertEqual(len(nb), len(set(nb)))
            self.assertTrue(set(nb) <= ids)
            dist = [n["distance"] for n in p["neighbours"]]
            self.assertEqual(dist, sorted(dist))
            self.assertTrue(all(0 <= n["stability"] <= 1 for n in p["neighbours"]))

    def test_every_player_has_one_subrole_in_their_group(self):
        roles = {s["id"]: s for s in self.d["subroles"]}
        for p in self.d["players"]:
            self.assertEqual(roles[p["subrole_id"]]["group"], p["group"])
        self.assertEqual(sum(s["count"] for s in self.d["subroles"]), len(self.d["players"]))

    def test_eligibility(self):
        mm = self.d["summary"]["min_minutes"]
        self.assertTrue(all(p["minutes"] >= mm for p in self.d["players"]))
        self.assertTrue(all(p["group"] != "Goalkeeper" for p in self.d["players"]))


if __name__ == "__main__":
    unittest.main()
