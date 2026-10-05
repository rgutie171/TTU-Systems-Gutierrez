"""Rogelio Gutierrez
Fewer Breaks in the 2026-2027 Champions League Calendar
"""
"""Question 2. Beating 48 Breaks
2B- Home / Optimizing

"""


import json
from pathlib import Path

import pandas as pd


FOLDER = Path(__file__).resolve().parent
INPUT = FOLDER / "original_calendar.csv"
FEDERATIONS = FOLDER / "ucl_club_federations.xlsx"
OUTPUT = FOLDER / "results"
TIME_LIMIT = 3600

WARM_START = None


def load_federations(path=FEDERATIONS):
    """Read exact club keys and consistent national-federation identifiers."""
    table = pd.read_excel(path, sheet_name="Club Federations", dtype=str)
    required = ["Club name", "National federation"]
    if not set(required).issubset(table.columns):
        raise ValueError(f"Federation spreadsheet must contain {required}")
    table = table[required]
    if table.isna().any().any() or table.apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Club names and national federations must not be blank")
    if table["Club name"].duplicated().any():
        raise ValueError("Federation spreadsheet contains duplicate club names")
    return dict(zip(table["Club name"], table["National federation"].str.strip()))


def same_federation_matches(calendar, federations):
    clubs = set(calendar.home_club) | set(calendar.away_club)
    missing = clubs - federations.keys()
    if missing:
        raise ValueError(f"Missing national federation for: {', '.join(sorted(missing))}")
    return calendar.index[
        calendar.home_club.map(federations) == calendar.away_club.map(federations)
    ].tolist()


def check_calendar(calendar, federations):
    """Count breaks and check feasibility using only calendar data."""
    prohibited = same_federation_matches(calendar, federations)
    if prohibited:
        fixtures = "; ".join(
            f"{calendar.loc[m, 'home_club']} vs {calendar.loc[m, 'away_club']}"
            for m in prohibited
        )
        raise ValueError(
            f"Same-federation matches are prohibited: {fixtures}. "
            "The fixed draw must be corrected; changing matchdays cannot fix these pairings."
        )
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
    
    assert 0 < TIME_LIMIT <= 3600
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_csv(INPUT).rename(columns={"home_team": "home_club", "away_team": "away_club"})
    calendar.insert(0, "match_id", range(1, len(calendar) + 1))
    calendar = calendar[["match_id", "matchday", "home_club", "away_club"]]
    federations = load_federations()
    prohibited_matches = same_federation_matches(calendar, federations)
    baseline = check_calendar(calendar, federations)
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

   
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("UCL_breaks_Q1a")
    model.Params.TimeLimit = TIME_LIMIT
    model.Params.Seed = 0
    model.Params.LogFile = str(OUTPUT / "gurobi.log")

    x = model.addVars(matches, days, vtype=GRB.BINARY, name="x")
    b = model.addVars(clubs, range(1, 8), lb=0, name="b")
    h = {}
    for club in clubs:
        for day in days:
            h[club, day] = gp.quicksum(x[m, day] for m in home_matches[club])

   
    for m in matches:
        model.addConstr(gp.quicksum(x[m, day] for day in days) == 1)


    for m in prohibited_matches:
        for day in days:
            model.addConstr(x[m, day] == 0, name=f"same_federation_{m}_{day}")

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

    model.setObjective(b.sum(), GRB.MINIMIZE)


    start_calendar = calendar if WARM_START is None else pd.read_csv(WARM_START)
    start_breaks = check_calendar(start_calendar, federations)
    fixed = ["match_id", "home_club", "away_club"]
    if calendar.match_id.duplicated().any() or start_calendar.match_id.duplicated().any():
        raise ValueError("Match IDs must be unique in the input and starting calendar")
    pd.testing.assert_frame_equal(
        calendar[fixed].sort_values("match_id").reset_index(drop=True),
        start_calendar[fixed].sort_values("match_id").reset_index(drop=True),
    )
    start_days = start_calendar.set_index("match_id").matchday
    for m in matches:
        for day in days:
            x[m, day].Start = int(start_days.loc[calendar.loc[m, "match_id"]] == day)

    for row in start_breaks.itertuples(index=False):
        pattern = row.pattern.split()
        for day in range(1, 8):
            b[row.club, day].Start = int(pattern[day - 1] == pattern[day])

    model.update()
    assert model.NumBinVars == 1152
    assert model.NumConstrs == 1368 + 8 * len(prohibited_matches)
    model.write(str(OUTPUT / "model.lp"))
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError("Gurobi did not return a feasible calendar.")

   
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

    results = check_calendar(exported, federations)
    total = int(results.breaks.sum())
    assert abs(total - model.ObjVal) < 1e-5
    results.to_csv(OUTPUT / "club_breaks.csv", index=False)


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
        "starting_breaks": int(start_breaks.breaks.sum()),
        "starting_calendar": str(INPUT if WARM_START is None else WARM_START),
        "binary_variables": model.NumBinVars,
        "constraints": model.NumConstrs,
        "independent_validation": {
            "matches_preserved": True,
            "no_same_federation_matches": True,
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
