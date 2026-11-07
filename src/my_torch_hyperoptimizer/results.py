import csv

from typeguard import typechecked

from src.my_torch_hyperoptimizer.search_space import SEARCH_SPACE
from src.my_torch_hyperoptimizer.trial import Trial

@typechecked
def writeResults(trials: list[Trial], path: str) -> None:
    """One row per configuration and per rung it reached, best first inside each rung."""
    rows = []
    for index, trial in enumerate(trials):
        for seen, score in trial.history:
            rows.append([index, seen, round(score, 2), *trial.config.values()])
    rows.sort(key=lambda row: (row[1], -row[2]))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "positions", "accuracy", *SEARCH_SPACE])
        writer.writerows(rows)
