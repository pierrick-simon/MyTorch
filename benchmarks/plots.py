"""Charts of the results of docs/BENCHMARKS.md, saved in docs/images/.

The numbers are copied from the tables of docs/BENCHMARKS.md: update both together.
Usage: python benchmarks/plots.py
"""
from os import makedirs, path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = path.join(path.dirname(path.dirname(path.abspath(__file__))), "docs", "images")

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D", "^"]
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]

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

# Speed: training step time of each network (with -O)
SPEED = [
    ("XOR", 9, 0.003),
    ("Hamming", 65, 0.011),
    ("Chess small", 25154, 1.963),
    ("Chess medium", 52322, 4.192),
    ("Chess large", 219331, 17.728),
]
SPEED_LABEL_OFFSETS = {"XOR": (8, -22, "left"), "Hamming": (-8, 8, "right"), "Chess small": (8, -24, "left"),
                       "Chess medium": (-10, 6, "right"), "Chess large": (-10, 4, "right")}

# Cost of type checking: training step with checks on / with -O
TYPEGUARD = [
    ("2-2-1", 16),
    ("6-8-1", 12.5),
    ("783-32-2", 8.7),
    ("783-64-32-2", 8.7),
    ("783-256-64-32-3", 8.9),
]

# Truth tables: seeds solved (out of 5) after 1,000 / 10,000 / 30,000 / 100,000 iterations
ITERATIONS = ["1,000", "10,000", "30,000", "100,000"]
CONVERGENCE = [
    ("AND 2-2-1, Sigmoid / Sigmoid", [0, 0, 5, 5]),
    ("XOR 2-2-1, Sigmoid / Sigmoid", [0, 0, 2, 5]),
    ("XOR 2-4-1, Sigmoid / Sigmoid", [0, 0, 3, 5]),
    ("XOR 2-4-1, Tanh / Sigmoid", [0, 4, 5, 5]),
    ("XOR 2-4-1, ReLU / Sigmoid", [0, 4, 5, 5]),
    ("XOR 2-4-1, LeakyReLU / Sigmoid", [0, 4, 5, 5]),
    ("Hamming 6-8-1, Sigmoid / Sigmoid", [0, 0, 0, 0]),
    ("Hamming 6-8-1, Sigmoid / Linear", [0, 2, 5, 5]),
    ("Hamming 6-8-1, Tanh / Linear", [0, 5, 5, 5]),
    ("Hamming 6-8-1, ReLU / Linear", [0, 4, 4, 5]),
    ("Hamming 6-16-1, Sigmoid / Linear", [0, 5, 5, 5]),
]

# Chess: 783-32-2, Sigmoid / Sigmoid, after 1 / 5 / 10 epochs
EPOCHS = [1, 5, 10]
CHESS_BASELINE = 65.8
CHESS = [
    ("Colours not distinguished, lr 0.01", [63.4, 65.4, 66.2], [94, 96, 96]),
    ("Colours distinguished, lr 0.002", [63.2, 64.3, 64.6], [93, 95, 95]),
    ("Colours distinguished, lr 0.01", [63.5, 65.7, 66.3], [90, 94, 95]),
    ("Colours distinguished, lr 0.05", [65.6, 67.3, 68.8], [94, 86, 79]),
]


def save(fig, name: str):
    fig.savefig(path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved docs/images/{name}")


def speed():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    params = [p for _, p, _ in SPEED]
    times = [t for _, _, t in SPEED]
    low, high = 5, 400000
    ax.plot([low, high], [low * 80e-6, high * 80e-6], color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    ax.text(3000, 3000 * 80e-6 * 1.6, "80 ns per parameter", color=TEXT_SECONDARY, rotation=29, fontsize=9)
    ax.plot(params, times, color=SERIES[0], linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=2)
    for name, p, t in SPEED:
        dx, dy, ha = SPEED_LABEL_OFFSETS[name]
        ax.annotate(f"{name}: {t:g} ms", (p, t), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Parameters (weights and biases)")
    ax.set_ylabel("Training step (ms, log scale)")
    ax.set_title("Training time grows linearly with the number of parameters")
    save(fig, "speed.png")


def typeguard():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = [n for n, _ in TYPEGUARD]
    slowdowns = [s for _, s in TYPEGUARD]
    bars = ax.barh(names, slowdowns, color=SERIES[0], height=0.6)
    ax.bar_label(bars, labels=[f"× {s:g}" for s in slowdowns], padding=4, color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 18)
    ax.set_xlabel("Training step without -O / with -O")
    ax.set_title("Type checking makes training 9 to 16 times slower")
    save(fig, "typeguard.png")


def convergence():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP, N=6)
    values = [solved for _, solved in CONVERGENCE]
    ax.imshow(values, cmap=cmap, vmin=0, vmax=5, aspect="auto")
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v}/5", ha="center", va="center", fontsize=9, color=SURFACE if v >= 3 else TEXT)
    ax.set_xticks(range(len(ITERATIONS)), ITERATIONS)
    ax.set_yticks(range(len(CONVERGENCE)), [n for n, _ in CONVERGENCE])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Iterations")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x + 0.5 for x in range(len(ITERATIONS) - 1)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(CONVERGENCE) - 1)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_title("Truth tables: seeds solved out of 5", pad=36)
    save(fig, "convergence.png")


def chess():
    fig, (acc, check) = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, (name, accuracies, checks) in enumerate(CHESS):
        style = {"color": SERIES[i], "linewidth": 2, "marker": MARKERS[i], "markersize": 8,
                 "markeredgecolor": SURFACE, "markeredgewidth": 1.5, "label": name}
        acc.plot(EPOCHS, accuracies, **style)
        check.plot(EPOCHS, checks, **style)
    acc.axhline(CHESS_BASELINE, color=TEXT_SECONDARY, linewidth=1, linestyle="--",
                label=f"Always answering Check ({CHESS_BASELINE} %)")
    acc.annotate(f"{CHESS[3][1][-1]} %", (EPOCHS[-1], CHESS[3][1][-1]), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    check.annotate(f"{CHESS[3][2][-1]} %", (EPOCHS[-1], CHESS[3][2][-1]), textcoords="offset points",
                   xytext=(0, -16), ha="center", fontsize=9, fontweight="bold")
    acc.set_title("Test accuracy (%)")
    check.set_title("Test positions predicted as Check (%)")
    for ax in (acc, check):
        ax.set_xticks(EPOCHS)
        ax.set_xlabel("Epochs")
    acc.set_ylim(62, 70)
    check.set_ylim(75, 100)
    handles, labels = acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Chess, 783-32-2: the colour fix and a larger learning rate beat the baseline",
                 x=0.06, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "chess.png")


if __name__ == "__main__":
    makedirs(OUT, exist_ok=True)
    speed()
    typeguard()
    convergence()
    chess()
