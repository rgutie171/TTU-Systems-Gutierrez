"""Rogelio Gutierrez
Fewer Breaks in the 2026-2027 Champions League Calendar
"""
"""Question 2. Beating 48 Breaks
2A- Home / Home Away Pattern File Creation
Number of breaks For Liverpool, Manchester City and Sporting CP
Confirm Total of Breaks - 48

"""

from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_FOLDER / "ucl_2026_27_uefa_calendar.csv"
OUTPUT_FILE = PROJECT_FOLDER / "ucl_break_analysis.xlsx"
REQUESTED_CLUBS = ["Liverpool", "Manchester City", "Sporting CP"]


def read_calendar(path):
    calendar = pd.read_csv(path, usecols=["matchday", "home_team", "away_team"])
    calendar = calendar.rename(columns={
        "matchday": "Matchday",
        "home_team": "Home Club",
        "away_team": "Away Club",
    })
    calendar = calendar[["Matchday", "Home Club", "Away Club"]]

    for column in ["Home Club", "Away Club"]:
        calendar[column] = calendar[column].str.strip()
        if calendar[column].isna().any() or (calendar[column] == "").any():
            raise ValueError(f"Missing club name in {column}.")

    return calendar


def analyze_breaks(calendar):
    if len(calendar) != 144:
        raise ValueError(f"Expected 144 matches; found {len(calendar)}.")

    patterns = {}
    opponents = {}

    for day, home, away in calendar.itertuples(index=False, name=None):
        for club in [home, away]:
            if club not in patterns:
                patterns[club] = {}
                opponents[club] = set()

        if home == away or away in opponents[home]:
            raise ValueError(f"Invalid or repeated pairing: {home} v {away}")

        opponents[home].add(away)
        opponents[away].add(home)

        for club, status in [(home, "H"), (away, "A")]:
            if day in patterns[club]:
                raise ValueError(f"{club} plays twice on matchday {day}.")
            patterns[club][day] = status

    if len(patterns) != 36:
        raise ValueError(f"Expected 36 clubs; found {len(patterns)}.")

    results = []
    for club in sorted(patterns):
        if set(patterns[club]) != set(range(1, 9)):
            raise ValueError(f"{club} does not play on all eight matchdays.")

        statuses = []
        for day in range(1, 9):
            statuses.append(patterns[club][day])

        if statuses.count("H") != 4 or statuses.count("A") != 4:
            raise ValueError(f"{club} must have four home and four away matches.")

               locations = []
        for i in range(7):
            if statuses[i] == statuses[i + 1]:
                matchdays = f"{i + 1}-{i + 2}"
                pattern = statuses[i] + statuses[i + 1]
                locations.append(f"{matchdays} ({pattern})")

        result = {"Club": club}
        for day in range(1, 9):
            result[f"MD{day}"] = statuses[day - 1]

        result["Pattern"] = " ".join(statuses)
        result["Break Locations"] = "; ".join(locations) or "None"
        result["Breaks"] = len(locations)
        results.append(result)

    return pd.DataFrame(results)


def save_results(calendar, all_clubs, path):
    requested = all_clubs.set_index("Club").loc[REQUESTED_CLUBS].reset_index()
    total = int(all_clubs["Breaks"].sum())

    
    total_row = pd.DataFrame([{"Club": "TOTAL", "Breaks": total}])
    summary = pd.concat([all_clubs, total_row], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        requested.to_excel(writer, sheet_name="Requested Clubs", index=False)
        summary.to_excel(writer, sheet_name="All Clubs", index=False)
        calendar.to_excel(writer, sheet_name="Calendar", index=False)

    print(requested[["Club", "Pattern", "Breaks"]].to_string(index=False))
    print(f"\nTotal: {total} breaks")
    print(f"Excel file saved to: {path.resolve()}")


if __name__ == "__main__":
    calendar = read_calendar(INPUT_FILE)
    all_clubs = analyze_breaks(calendar)
    save_results(calendar, all_clubs, OUTPUT_FILE)
