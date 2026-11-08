import os
import subprocess
import sys

from typeguard import typechecked

from src.my_torch_hyperoptimizer.constants import (
    DATASET_PATH,
    TRIALS_DIR,
    VALIDATION_PATH,
)
from src.my_torch_hyperoptimizer.trial import Trial


@typechecked
def runBenchmark(script: str, args: list[str]) -> None:
    """Run a script of benchmarks/ with the same Python, hiding its output but not its errors."""
    command = [sys.executable, "-O", os.path.join("benchmarks", script), *args]
    if subprocess.run(command, stdout=subprocess.DEVNULL, check=False).returncode != 0:
        print(f"my_torch_hyperoptimizer: benchmarks/{script} failed", file=sys.stderr)

@typechecked
def plotCheckpoints(trials: list[Trial], validationSize: int) -> None:
    """Evaluate the checkpoints of every trial on the validation set with benchmarks/checkpoints_csv.py, then draw
    with benchmarks/checkpoints_plot.py one chart per rung, with the trials that reached it: each curve stops
    where its trial was eliminated."""
    with open(DATASET_PATH) as f:  # the validation set of loadData is the start of the dataset
        lines = f.readlines()[:validationSize]
    with open(VALIDATION_PATH, "w") as f:
        f.writelines(lines)

    for i, trial in enumerate(trials, 1):
        print(f"Evaluating the checkpoints of {trial.directory} ({i}/{len(trials)})")
        runBenchmark("checkpoints_csv.py", [trial.directory, VALIDATION_PATH])

    for rung in range(1, max(len(trial.history) for trial in trials) + 1):
        csvs = [os.path.join(t.directory, "checkpoints.csv") for t in trials if len(t.history) >= rung]
        output = os.path.join(TRIALS_DIR, f"rung_{rung}.png")
        runBenchmark("checkpoints_plot.py", [csv for csv in csvs if os.path.isfile(csv)] + ["-o", output])
        print(f"Curves of the {len(csvs)} configurations of rung {rung} in {output}")
