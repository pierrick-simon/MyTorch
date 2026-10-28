"""Accuracy and training time of several networks on the truth tables of my_torch_bonus.

Usage: python -O benchmarks/tables.py [ITERATIONS]
"""
import random
import sys
import time
from importlib import import_module
from os import path

sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))
from src.my_torch.activationClasses import ActivationClassNames
from src.my_torch.MLP import MLP

TOLERANCES = {"and": 0.1, "xor": 0.1, "hamming": 0.5}
CONFIGS = [
    ("and", [2], 0.1, "Sigmoid", "Sigmoid"),
    ("xor", [2], 0.1, "Sigmoid", "Sigmoid"),
    ("xor", [4], 0.1, "Sigmoid", "Sigmoid"),
    ("xor", [4], 0.1, "Tanh", "Sigmoid"),
    ("xor", [4], 0.1, "ReLU", "Sigmoid"),
    ("xor", [4], 0.1, "LeakyReLU", "Sigmoid"),
    ("hamming", [8], 0.05, "Sigmoid", "Sigmoid"),
    ("hamming", [8], 0.05, "Sigmoid", "Linear"),
    ("hamming", [8], 0.05, "Tanh", "Linear"),
    ("hamming", [8], 0.01, "ReLU", "Linear"),
    ("hamming", [16], 0.05, "Sigmoid", "Linear"),
]
SEEDS = [1, 2, 3, 4, 5]
ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 100000


def accuracy(mlp: MLP, table: list, tolerance: float) -> float:
    """Share of the rows of the table where every output is within the tolerance."""
    ok = 0
    for case in table:
        outputs = mlp.tick(case["inputs"])
        ok += all(abs(o - e) < tolerance for o, e in zip(outputs, case["expectedOutputs"]))
    return ok / len(table)


print(f"{ITERATIONS} iterations, seeds {SEEDS}")
print("| Task | Shape | Learning rate | Hidden / output | Mean accuracy | Seeds solved | Time per training (s) |")
print("|---|---|--:|---|--:|--:|--:|")
for task, layers, lr, hidden, output in CONFIGS:
    table = import_module(f"src.bonus.tables.{task}").TABLE
    nbInputs = len(table[0]["inputs"])
    accuracies, times = [], []
    for seed in SEEDS:
        mlp = MLP(nbInputs, layers, 1, lr, (ActivationClassNames[hidden], ActivationClassNames[output]), seed)
        random.seed(seed)
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            case = table[random.randint(0, len(table) - 1)]
            mlp.learn(mlp.tick(case["inputs"]), case["expectedOutputs"])
        times.append(time.perf_counter() - start)
        accuracies.append(accuracy(mlp, table, TOLERANCES[task]))
    shape = "-".join(map(str, [nbInputs, *layers, 1]))
    solved = sum(a == 1.0 for a in accuracies)
    mean = sum(accuracies) / len(accuracies) * 100
    print(f"| {task} | {shape} | {lr} | {hidden} / {output} | {mean:.1f} % | {solved}/{len(SEEDS)} | {sum(times) / len(times):.1f} |", flush=True)
