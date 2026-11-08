"""Measures the alternatives of docs/ALTERNATIVE_PROPOSAL.md with a NumPy prototype.

The prototype is the best network of the hyperoptimizer (141-256-64-5, LeakyReLU / Sigmoid,
learning rate 0.0115, momentum 0.7996, initial bias 0.187, He / Xavier initialisation, balanced
epochs), trained and validated on the same data as the hyperoptimizer (hyperopt/dataset_1000.txt).

Usage (from the root of the repository):
    OMP_NUM_THREADS=1 .venv/bin/python -O benchmarks/alternatives.py data
"""
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.my_torch_analyzer.fen_parser import parse_line
from src.my_torch_hyperoptimizer.constants import DATASET_PATH, VALIDATION_RATIO
from src.my_torch_hyperoptimizer.dataset import loadData

LAYERS = [141, 256, 64, 5]
LR, MOMENTUM, BIAS = 0.0115, 0.7996, 0.187
MAX_OVERSAMPLING = 4
SEEDS = 3


def toArrays(boards: list) -> tuple[np.ndarray, np.ndarray]:
    return np.array([c.getInput() for c in boards]), np.array([c.getSate() for c in boards])


def balancedEpoch(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    byClass = [np.where(labels == k)[0] for k in range(LAYERS[-1])]
    target = max(len(items) for items in byClass)
    epoch = []
    for items in byClass:
        epoch.extend(items)
        epoch.extend(rng.choice(items, min(target, MAX_OVERSAMPLING * len(items)) - len(items)))
    epoch = np.array(epoch)
    rng.shuffle(epoch)
    return epoch


def initNetwork(rng: np.random.Generator) -> tuple[list, list]:
    weights, biases = [], []
    for i in range(len(LAYERS) - 1):
        nbIn, nbOut = LAYERS[i], LAYERS[i + 1]
        hidden = i < len(LAYERS) - 2
        limit = np.sqrt(6 / nbIn) if hidden else np.sqrt(6 / (nbIn + nbOut))
        weights.append(rng.uniform(-limit, limit, (nbIn, nbOut)))
        biases.append(np.full(nbOut, BIAS))
    return weights, biases


def forward(weights: list, biases: list, x: np.ndarray) -> list:
    activations = [x]
    for i in range(len(weights)):
        z = activations[-1] @ weights[i] + biases[i]
        if i < len(weights) - 1:
            a = np.where(z > 0, z, 0.01 * z)
        else:
            a = 1 / (1 + np.exp(-z))
        activations.append(a)
    return activations


def train(data: tuple, validation: tuple, budget: int, seed: int, batch: int = 1, lr: float = LR, momentum: float = MOMENTUM) -> float:
    """Trains on `budget` positions and returns the validation accuracy in %."""
    x, y = data
    labels = y.argmax(1)
    rng = np.random.default_rng(seed)
    weights, biases = initNetwork(rng)
    velocity = [np.zeros_like(w) for w in weights]
    seen = 0
    while seen < budget:
        epoch = balancedEpoch(labels, rng)
        for start in range(0, len(epoch), batch):
            idx = epoch[start:start + batch]
            activations = forward(weights, biases, x[idx])
            out = activations[-1]
            # same convention as MyTorch: error = -dLoss/dz, then w += lr * error * input
            error = (y[idx] - out) * out * (1 - out)
            for i in reversed(range(len(weights))):
                previous = (error @ weights[i].T) * np.where(activations[i] > 0, 1, 0.01) if i > 0 else None
                velocity[i] = lr * activations[i].T @ error / len(idx) + momentum * velocity[i]
                weights[i] += velocity[i]
                biases[i] += lr * error.mean(0)
                error = previous
            seen += len(idx)
            if seen >= budget:
                break
    predictions = forward(weights, biases, validation[0])[-1].argmax(1)
    return float((predictions == validation[1].argmax(1)).mean() * 100)


def report(name: str, data: tuple, validation: tuple, **kwargs) -> None:
    start = time.perf_counter()
    scores = [train(data, validation, seed=seed, **kwargs) for seed in range(SEEDS)]
    elapsed = (time.perf_counter() - start) / SEEDS
    print(f"{name}: {np.mean(scores):.2f} % (min {min(scores):.2f}, max {max(scores):.2f}), {elapsed:.1f} s per training", flush=True)


def largerData(perFile: int) -> tuple:
    """perFile positions from each of the 9 files of dataset/, none of them in the validation set."""
    lines = Path(DATASET_PATH).read_text().splitlines()
    validationLines = set(lines[:int(len(lines) * VALIDATION_RATIO)])
    rng = random.Random(0)
    picked = []
    for path in sorted(glob.glob("dataset/*/*.txt")):
        candidates = [line for line in Path(path).read_text().splitlines() if line.strip() and line not in validationLines]
        picked += rng.sample(candidates, min(perFile, len(candidates)))
    return toArrays([parse_line(line) for line in picked])


def main() -> None:
    trainBoards, validationBoards = loadData()
    data, validation = toArrays(trainBoards), toArrays(validationBoards)
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "data":
        report(f"{len(data[0])} training positions, batch 32, 900,000 positions", data, validation, budget=900000, batch=32, lr=0.05, momentum=0.9)
        larger = largerData(8000)
        print(f"{len(larger[0])} training positions")
        for budget in (900000, 3000000):
            report(f"batch 32, {budget:,} positions", larger, validation, budget=budget, batch=32, lr=0.05, momentum=0.9)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
