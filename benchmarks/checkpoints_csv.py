"""Evaluate the checkpoints saved by my_torch_analyzer --train and write the results in a CSV.

The analyzer saves model_N.nn every CHECKPOINT_EVERY positions in the directory named after SAVEFILE.
Every checkpoint is evaluated on the same positions, so the results can be compared.
Usage: python -O benchmarks/checkpoints_csv.py DIRECTORY [CHESSFILE] [PER_FILE]
- CHESSFILE: labelled positions to evaluate on. Default: PER_FILE positions picked from each file of dataset/.
- PER_FILE: positions per file of dataset/ when no CHESSFILE is given. Default: 100 (900 positions).
The CSV is saved in DIRECTORY/checkpoints.csv with the columns: iteration,accuracy,loss
(accuracy in %, loss = mean squared error). A best.nn file, if present, is added as the last row with
iteration "best".
Note: a file that is not a .nn inside DIRECTORY may make the next --train refuse to overwrite the directory.
"""
import csv
import random
import re
import sys
from os import listdir, path

ROOT = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.my_torch.MLP import MLP
from src.my_torch_analyzer.chessboard import Chessboard
from src.my_torch_analyzer.fen_parser import fen_parser, parse_line

FILES = [f"{c}/{p}_pieces.txt" for c in ("nothing", "check", "checkmate") for p in ("10", "20", "many")]
CHECKPOINT = re.compile(r"model_(\d+)\.nn")
CSV_NAME = "checkpoints.csv"


def sampleDataset(perFile: int) -> list[Chessboard]:
    rng = random.Random(42)
    lines = []
    for f in FILES:
        with open(path.join(ROOT, "dataset", f)) as fd:
            content = fd.readlines()
        lines += rng.sample(content, min(perFile, len(content)))
    return [parse_line(line) for line in lines]


def evaluate(mlp: MLP, data: list[tuple[list[float], list[float]]]) -> tuple[float, float]:
    """Accuracy (%) and mean squared error, the loss minimised by MLP.learn."""
    ok = 0
    loss = 0.0
    for inputs, target in data:
        output = mlp.tick(inputs)
        ok += output.index(max(output)) == target.index(max(target))
        loss += sum((t - o) ** 2 for o, t in zip(output, target, strict=True)) / len(target)
    return ok / len(data) * 100, loss / len(data)


def main() -> int:
    if len(sys.argv) < 2 or not path.isdir(sys.argv[1]):
        print(__doc__, file=sys.stderr)
        return 84
    directory = sys.argv[1]
    checkpoints = sorted((int(m.group(1)), name) for name in listdir(directory) if (m := CHECKPOINT.fullmatch(name)))
    if not checkpoints:
        print(f"No model_N.nn file in {directory}", file=sys.stderr)
        return 84

    if len(sys.argv) > 2 and not sys.argv[2].isdigit():
        boards = fen_parser(sys.argv[2])
    else:
        boards = sampleDataset(int(sys.argv[-1]) if sys.argv[-1].isdigit() else 100)
    if any(not b.hasState() for b in boards):
        print("Every position must have a label", file=sys.stderr)
        return 84
    data = [(b.getInput(), b.getSate()) for b in boards]
    print(f"{len(checkpoints)} checkpoints, evaluated on {len(data)} positions")

    rows = []
    for iteration, name in checkpoints:
        accuracy, loss = evaluate(MLP.load(path.join(directory, name), False), data)
        rows.append((iteration, accuracy, loss))
        print(f"{iteration}: {accuracy:.1f} % / {loss:.4f}", flush=True)

    best = path.join(directory, "best.nn")
    if path.isfile(best):
        accuracy, loss = evaluate(MLP.load(best, False), data)
        rows.append(("best", accuracy, loss))
        print(f"best.nn: {accuracy:.1f} % / {loss:.4f}")

    output = path.join(directory, CSV_NAME)
    with open(output, "w", newline="") as fd:
        writer = csv.writer(fd)
        writer.writerow(["iteration", "accuracy", "loss"])
        for iteration, accuracy, loss in rows:
            writer.writerow([iteration, f"{accuracy:.4f}", f"{loss:.6f}"])
    print(f"CSV saved in {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())