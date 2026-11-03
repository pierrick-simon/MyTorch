"""Accuracy and loss of the checkpoints saved by my_torch_analyzer --train, on the same small dataset.

The analyzer saves model_N.nn every CHECKPOINT_EVERY positions in the directory named after SAVEFILE.
Every checkpoint is evaluated on the same positions, so the curves can be compared.
Usage: python -O benchmarks/checkpoints.py DIRECTORY [CHESSFILE] [PER_FILE]
- CHESSFILE: labelled positions to evaluate on. Default: PER_FILE positions picked from each file of dataset/.
- PER_FILE: positions per file of dataset/ when no CHESSFILE is given. Default: 100 (900 positions).
The chart is saved next to the directory, in DIRECTORY_checkpoints.png: a file that is not a .nn inside it
would make the next --train refuse to overwrite the directory.
"""
import random
import re
import sys
from os import listdir, path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.my_torch.MLP import MLP
from src.my_torch_analyzer.chessboard import Chessboard
from src.my_torch_analyzer.fen_parser import fen_parser, parse_line

FILES = [f"{c}/{p}_pieces.txt" for c in ("nothing", "check", "checkmate") for p in ("10", "20", "many")]
CHECKPOINT = re.compile(r"model_(\d+)\.nn")

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
})


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

    iterations, accuracies, losses = [], [], []
    print("| Iteration | Accuracy | Loss |")
    print("|--:|--:|--:|")
    for iteration, name in checkpoints:
        accuracy, loss = evaluate(MLP.load(path.join(directory, name), False), data)
        iterations.append(iteration)
        accuracies.append(accuracy)
        losses.append(loss)
        print(f"| {iteration} | {accuracy:.1f} % | {loss:.4f} |", flush=True)

    best = path.join(directory, "best.nn")
    bestAccuracy = None
    if path.isfile(best):
        bestAccuracy, bestLoss = evaluate(MLP.load(best, False), data)
        print(f"| best.nn | {bestAccuracy:.1f} % | {bestLoss:.4f} |")

    fig, (accAx, lossAx) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    accAx.plot(iterations, accuracies, color=SERIES[0], linewidth=2, marker="o", markersize=3)
    if bestAccuracy is not None:
        accAx.axhline(bestAccuracy, color=TEXT_SECONDARY, linewidth=1, linestyle="--", label=f"best.nn ({bestAccuracy:.1f} %)")
        accAx.legend(loc="lower right")
    accAx.set_title("Accuracy of the checkpoints")
    accAx.set_ylabel("Accuracy (%)")
    accAx.set_ylim(0, 100)
    lossAx.plot(iterations, losses, color=SERIES[1], linewidth=2, marker="o", markersize=3)
    lossAx.set_title("Loss of the checkpoints (mean squared error)")
    lossAx.set_ylabel("Loss")
    lossAx.set_xlabel("Training iterations (positions seen)")
    fig.text(0.01, 0.005, f"{len(data)} positions, same for every checkpoint", color=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    output = f"{path.normpath(directory)}_checkpoints.png"
    fig.savefig(output, dpi=150)
    print(f"Chart saved in {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
