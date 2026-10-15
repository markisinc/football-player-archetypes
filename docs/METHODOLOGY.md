# Methodology: player style archetypes & similarity

Built on **StatsBomb Open Data 2015/16** for the Premier League, La Liga, Serie A and Ligue 1. It does two things:

1. **Archetypes**: 21 styles, found within 5 position groups.
2. **Similar players**: pick a player and get the closest styles, with a score and a "holds up" signal for each match.

Similarity describes the recorded style of play in 2015/16. It does not measure ability, value or whether one player could replace another.

## Run it

See the [README](../README.md#run-it-yourself). `src/sb_profiling.py` writes `outputs/profiling_sb/` and refreshes `docs/profiles.json`, which the live explorer reads.

## Results (30 whole-match resamples)

| | |
|---|---|
| Matches / eligible players | 1,517 / **1,267** outfield players with 900+ minutes |
| Search space | 27 style measures → 8 PCA dimensions holding 87% of the variation |
| Archetypes | Centre-back 3, Full-back 5, Midfielder 5, Attacking mid/winger 4, Forward 4 |
| Archetype agreement across resamples (ARI) | CB 0.69 · MID 0.60 · AM/W 0.50 · FB 0.46 · FW 0.39 |
| Players who are core members (same archetype in ≥80% of resamples) | 56%. 12% are borderline (<50%) |
| Top-5 neighbours surviving a resample | 37% on average; 16% of top-5 links are "stable" (≥60% of resamples) |
| Top-5 neighbours from the same league | 29% (chance: 25%) |
| Top-5 neighbours from the same team | 2.3% (chance: 1.2%) |
| Lineups flagged for review | 34 of 3,034 team-matches |

**How to read this:** the archetypes are real tendencies, but within a position styles blend together. Treat them as guides, and use each player's core/borderline badge. For exact neighbour lists, trust the "holds up" badge rather than rank alone. With roughly 30 matches per player-season, the per-90 rates carry real sampling noise.

## Design decisions

| Area | Simpler approach | What this project does |
|---|---|---|
| Minutes | Infer from first/last touch | From official lineups (on/off clock times). Data quirks handled: starters mislabelled "Tactical Shift", spells after a player's own substitution, overlapping injury on/off spells (minutes = union of spells) |
| xG | Fit a location-only model | StatsBomb xG per shot |
| Features | Mix style and success rates | 27 **style-only** measures. Success rates, goals and xG are shown as context, not used for similarity |
| Measures | Basic counts | Receptions, carries, progressive passes and carries, long/forward pass share, crosses (open play), passes leading to shots, pressures, blocks, height of defensive actions |
| Defending | Raw counts | **Possession-adjusted**: count × 0.5 / opponent possession share (clipped 0.6–1.6) |
| Archetypes | One k-means over everyone (rediscovers positions) | **Two levels**: position group from lineup minutes, then k-means within each group |
| Picking k | Best silhouette (always picks the coarsest split) | The most detailed k (3–5) that stays reproducible: seed ARI ≥ 0.8 and every archetype ≥ 15 players |
| Neighbour stability | One overall number, or none | Per neighbour: share of resamples where they stay in the top 5 (25 nearest tracked) |

## Feature definitions

Pitch rescaled to 0–100 in both directions; every team attacks left → right. Per 90 minutes unless marked.

- **Volume:** passes, ball receptions, carries, take-ons attempted.
- **Passing style:** long-pass share (≥32 units on the 120-unit pitch) and forward-pass share (≥10 yards forward) of open-play passes; progressive passes (completed open-play passes from outside the defensive 30% that cut the distance to goal by ≥25%); open-play crosses; passes leading to shots.
- **Carrying:** progressive carries (same 25% rule, at least 4 units long).
- **Location** (share of on-ball actions: receptions, passes, take-ons, shots, recoveries, interceptions, clearances, blocks, duels, miscontrols, dispossessions): average depth, final-third share, opponent-box share, own-box share, average distance from the centre line, lateral spread.
- **Defending:** pressures, tackles, interceptions, recoveries, clearances and blocks (all possession-adjusted); aerial duels; average height of defensive actions; fouls committed.
- **Shooting:** non-penalty shots and non-penalty xG per shot (chance quality as a shooting-style choice).
- **Context only:** pass completion, take-on, aerial and tackle success (smoothed toward the cohort rate; no attempts = unknown), goals, npxG per 90, dispossessions, miscontrols, team possession.

Modelling: median-impute, cap at the 1st/99th percentiles, z-score, PCA ≥85%. Similarity = Euclidean distance across all retained dimensions. The explorer's **Match %** = 100 × (1 − distance / median distance between two random players). Within each group, archetypes use that group's own scaling and PCA, so they reflect what separates players *within* the position.

**Naming:** archetype names were chosen after reading each cluster's highest and lowest features and example players. They are matched to clusters by those defining features (one-to-one within a group), so a rerun keeps sensible names. They describe event patterns, not tactical instructions.

| Group | Archetypes |
|---|---|
| Centre-back | Ball-playing centre-back · Box-defending stopper · Wide centre-back (back three) |
| Full-back | Touchline progressor · Overlapping attacking full-back · Possession full-back · Conservative full-back · Aggressive defending full-back |
| Midfielder | Deep controller · Long-passing creator · Holding screen · Progressive box-to-box · Box-crashing midfielder |
| Attacking mid / winger | Central creative playmaker · Hard-working wide midfielder · Inside forward / goal threat · Dribbling winger |
| Forward | Dribbling channel-runner · Wide-roaming worker · Penalty-box target striker · Mobile front-line finisher |

## Outputs (`outputs/profiling_sb/`)

| File | Contents |
|---|---|
| `profiles.json` | Everything the explorer needs: players, archetypes, neighbours with stability, map coordinates, heatmaps, loadings, summary |
| `player_profiles.csv` | Player-season features, context, group, archetype |
| `subroles.csv` | Each archetype: size, defining features, examples |
| `subrole_candidates.csv` | k=3–5 per group: silhouette, seed ARI, smallest size, which were eligible |
| `bootstrap_stability.csv` | Per resample: top-5 overlap and archetype ARI per group |
| `minutes_audit.csv` | Per team-match: starters, min/max players on the pitch, flag |
| `pca_loadings.csv`, `feature_correlations.csv`, `group_subrole_counts.csv`, `model_transform.json` | Diagnostics |

## Limits

- One season. Event data can't see off-ball movement, instructions or teammates' influence beyond what the actions show.
- Position groups come from StatsBomb's lineup positions. A player who split time is placed in the group where he played most (share recorded as `group_share`).
- League and team effects are small but present (29% vs 25% same-league neighbours). Use the explorer's league filter when that matters.
- Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Credit StatsBomb when you publish anything built on it.
