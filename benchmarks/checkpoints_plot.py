"""Plot the accuracy and loss curves of one or several CSV files made by checkpoints_csv.py.

Usage: python benchmarks/plot_checkpoints.py CSV [CSV ...] [-o OUTPUT]
- CSV: file(s) with the columns iteration,accuracy,loss. One curve per file, labelled with the name of the
  directory containing the file (or the file name if the directory name is already used by another curve).
  The "best" row, if present, is drawn as a dashed horizontal line of the same colour.
- -o OUTPUT: image to write. Default: checkpoints_comparison.png in the current directory.
"""
import argparse
import csv
import sys
from os import path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#2ba35f", "#8a4fd3", "#d1a21b", "#d6407f", "#3aa6b8", "#7a7a72"]

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


class Curve:
    def __init__(self, label: str):
        self.label = label
        self.iterations: list[int] = []
        self.accuracies: list[float] = []
        self.losses: list[float] = []
        self.best: tuple[float, float] | None = None


def readCsv(file: str) -> Curve:
    curve = Curve(file)
    with open(file, newline="") as fd:
        for row in csv.DictReader(fd):
            accuracy, loss = float(row["accuracy"]), float(row["loss"])
            if row["iteration"] == "best":
                curve.best = (accuracy, loss)
                continue
            curve.iterations.append(int(row["iteration"]))
            curve.accuracies.append(accuracy)
            curve.losses.append(loss)
    if not curve.iterations:
        raise ValueError("no checkpoint row")
    return curve


def labels(files: list[str]) -> list[str]:
    """Directory name of each file, or the file path when two files share the same directory name."""
    names = [path.basename(path.dirname(path.abspath(f))) for f in files]
    return [n if names.count(n) == 1 else f for n, f in zip(names, files, strict=True)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot accuracy and loss curves from checkpoint CSV files.")
    parser.add_argument("csv", nargs="+", help="CSV file(s) made by checkpoints_csv.py")
    parser.add_argument("-o", "--output", default="checkpoints_comparison.png", help="output image")
    args = parser.parse_args()

    curves = []
    for file, label in zip(args.csv, labels(args.csv), strict=True):
        try:
            curve = readCsv(file)
        except (OSError, KeyError, ValueError) as e:
            print(f"{file}: {e}", file=sys.stderr)
            return 84
        curve.label = label
        curves.append(curve)

    fig, (accAx, lossAx) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for i, c in enumerate(curves):
        color = SERIES[i % len(SERIES)]
        accAx.plot(c.iterations, c.accuracies, color=color, linewidth=2, marker="o", markersize=3, label=c.label)
        lossAx.plot(c.iterations, c.losses, color=color, linewidth=2, marker="o", markersize=3, label=c.label)
        if c.best is not None:
            accAx.axhline(c.best[0], color=color, linewidth=1, linestyle="--", alpha=0.6)
            lossAx.axhline(c.best[1], color=color, linewidth=1, linestyle="--", alpha=0.6)
    accAx.set_title("Accuracy of the checkpoints")
    accAx.set_ylabel("Accuracy (%)")
    accAx.set_ylim(0, 100)
    lossAx.set_title("Loss of the checkpoints (mean squared error)")
    lossAx.set_ylabel("Loss")
    lossAx.set_xlabel("Training iterations (positions seen)")
    if len(curves) > 1:
        accAx.legend(loc="lower right")
    if any(c.best is not None for c in curves):
        fig.text(0.01, 0.005, "Dashed lines: best.nn", color=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"Chart saved in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())