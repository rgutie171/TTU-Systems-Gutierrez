"""UCL scheduling with strict city-pair constraints.

Install: pip install gurobipy pandas openpyxl
Run:     python solve_ucl_city.py

Configure your Gurobi license locally before running.
"""

import json
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


# Change these settings if needed. The time limit must not exceed one hour.
FOLDER = Path(__file__).resolve().parent
INPUT = FOLDER / "original_calendar.csv"
OUTPUT = FOLDER / "results"
TIME_LIMIT = 3600
CITY_PAIRS = [("Real Madrid", "Atletico de Madrid"),
              ("Manchester City", "Manchester United"),
              ("Fenerbahce", "Galatasaray"), ("Lille", "Lens")]


def check_calendar(calendar):
    """Count breaks and check feasibility using only calendar data."""
    clubs = sorted(set(calendar.home_club) | set(calendar.away_club))
    assert len(calendar) == 144 and len(clubs) == 36
    results = []

    for club in clubs:
        pattern = ""
        for day in range(1, 9):
            games = calendar[
                (calendar.matchday == day)
                & ((calendar.home_club == club) | (calendar.away_club == club))
            ]
            assert len(games) == 1, f"{club}: incorrect match count on day {day}"
            pattern += "H" if games.iloc[0].home_club == club else "A"

        assert pattern.count("H") == 4, f"{club}: must have four home matches"
        assert "HHH" not in pattern and "AAA" not in pattern, club
        breaks = sum(pattern[d] == pattern[d + 1] for d in range(7))
        results.append({
            "club": club,
            "pattern": " ".join(pattern),
            "breaks": breaks,
        })

    return pd.DataFrame(results)


def main():
    # 1. Read and check the supplied calendar.
    assert 0 < TIME_LIMIT <= 3600
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_csv(INPUT).rename(columns={"home_team": "home_club", "away_team": "away_club"})
    calendar.insert(0, "match_id", range(1, len(calendar) + 1))
    calendar = calendar[["match_id", "matchday", "home_club", "away_club"]]
    baseline = check_calendar(calendar)
    assert baseline.breaks.sum() == 48

    clubs = baseline.club.tolist()
    matches = calendar.index.tolist()
    days = range(1, 9)
    home_matches = {}
    club_matches = {}
    for club in clubs:
        home_matches[club] = calendar.index[calendar.home_club == club].tolist()
        club_matches[club] = calendar.index[
            (calendar.home_club == club) | (calendar.away_club == club)
        ].tolist()

    # 2. Create the variables and home indicators from Question 1(a).
    model = gp.Model("UCL_breaks_strict_city")
    model.Params.TimeLimit = TIME_LIMIT
    model.Params.Seed = 0
    model.Params.LogFile = str(OUTPUT / "gurobi.log")

    x = model.addVars(matches, days, vtype=GRB.BINARY, name="x")
    b = model.addVars(clubs, range(1, 8), lb=0, name="b")
    h = {}
    for club in clubs:
        for day in days:
            h[club, day] = gp.quicksum(x[m, day] for m in home_matches[club])

    # 3. Add the scheduling and break-counting constraints.
    for m in matches:
        model.addConstr(gp.quicksum(x[m, day] for day in days) == 1)

    for club in clubs:
        for day in days:
            model.addConstr(
                gp.quicksum(x[m, day] for m in club_matches[club]) == 1
            )

    for club in clubs:
        for day in range(1, 7):
            three_days = h[club, day] + h[club, day + 1] + h[club, day + 2]
            model.addConstr(three_days <= 2)
            model.addConstr(three_days >= 1)

        for day in range(1, 8):
            model.addConstr(b[club, day] >= h[club, day] + h[club, day + 1] - 1)
            model.addConstr(b[club, day] >= 1 - h[club, day] - h[club, day + 1])

    # The paired clubs cannot both host on the same matchday.
    for first, second in CITY_PAIRS:
        for day in days:
            model.addConstr(h[first, day] + h[second, day] <= 1)

    model.setObjective(b.sum(), GRB.MINIMIZE)

    # 4. Offer the baseline as a repair candidate; it violates the city rule.
    for m in matches:
        for day in days:
            x[m, day].Start = int(calendar.loc[m, "matchday"] == day)

    for row in baseline.itertuples(index=False):
        pattern = row.pattern.split()
        for day in range(1, 8):
            b[row.club, day].Start = int(pattern[day - 1] == pattern[day])

    model.update()
    assert model.NumBinVars == 1152 and model.NumConstrs == 1400
    model.write(str(OUTPUT / "model.lp"))
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError("Gurobi did not return a feasible calendar.")

    # 5. Export the best calendar and independently check the exported CSV.
    best = calendar.copy()
    for m in matches:
        assigned_days = [day for day in days if x[m, day].X > 0.5]
        assert len(assigned_days) == 1
        best.loc[m, "matchday"] = assigned_days[0]

    best = best.sort_values(["matchday", "match_id"])
    best.to_csv(OUTPUT / "best_calendar.csv", index=False)
    exported = pd.read_csv(OUTPUT / "best_calendar.csv")

    fixed_columns = ["match_id", "home_club", "away_club"]
    original_matches = calendar[fixed_columns].sort_values("match_id").reset_index(drop=True)
    exported_matches = exported[fixed_columns].sort_values("match_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(original_matches, exported_matches)

    for day in days:
        hosts = set(exported.loc[exported.matchday == day, "home_club"])
        for first, second in CITY_PAIRS:
            assert not (first in hosts and second in hosts), (day, first, second)

    results = check_calendar(exported)
    total = int(results.breaks.sum())
    assert abs(total - model.ObjVal) < 1e-5
    results.to_csv(OUTPUT / "club_breaks.csv", index=False)

    # 6. Save the same solver statistics and supporting files.
    statistics = {
        "gurobi_version": gp.gurobi.version(),
        "time_limit_seconds": TIME_LIMIT,
        "runtime_seconds": model.Runtime,
        "status": model.Status,
        "best_breaks": total,
        "lower_bound": model.ObjBound,
        "gap": model.MIPGap,
        "nodes": model.NodeCount,
        "baseline_breaks": 48,
        "binary_variables": model.NumBinVars,
        "constraints": model.NumConstrs,
        "independent_validation": {
            "strict_city_rule": True,
            "matches_preserved": True,
            "once_per_matchday": True,
            "no_three_consecutive": True,
            "four_home_four_away": True,
            "break_count_matches_solver": True,
        },
    }
    (OUTPUT / "solver_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    model.write(str(OUTPUT / "best_solution.sol"))
    print(json.dumps(statistics, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
