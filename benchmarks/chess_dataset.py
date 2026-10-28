"""Accuracy of a 783-32-2 network on a balanced sample of the chess dataset.

Usage: python -O benchmarks/chess_dataset.py [EPOCHS] [LEARNING_RATE]
Requires the dataset: unzip dataset.zip
"""
import random
import sys
import time
from os import path

ROOT = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.my_torch.activationClasses import Sigmoid
from src.my_torch.MLP import MLP
from src.my_torch_analyzer.fen_parser import parse_line

FILES = [f"{c}/{p}_pieces.txt" for c in ("nothing", "check", "checkmate") for p in ("10", "20", "many")]
PER_FILE = 1000  # 9000 positions: 1/3 Nothing, 2/3 Check or Checkmate
TRAIN_SIZE = 7200
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01


def predicted(mlp: MLP, inputs: list[float]) -> int:
    outputs = mlp.tick(inputs)
    return outputs.index(max(outputs))


random.seed(42)
lines = []
for f in FILES:
    with open(path.join(ROOT, "dataset", f)) as fd:
        lines += random.sample(fd.readlines(), PER_FILE)
start = time.perf_counter()
boards = [parse_line(line) for line in lines]
duration = time.perf_counter() - start
print(f"Parsed {len(boards)} positions in {duration:.2f} s ({len(boards) / duration:.0f} positions/s)")

data = [(b.getInput(), b.getSate().index(1)) for b in boards]
random.shuffle(data)
train, test = data[:TRAIN_SIZE], data[TRAIN_SIZE:]
print(f"Share of Check in the test set: {sum(t for _, t in test) / len(test) * 100:.1f} %")

mlp = MLP(783, [32], 2, LR, (Sigmoid, Sigmoid), 1)
print("| Epoch | Train accuracy | Test accuracy | Test predicted as Check | Time (s) |")
print("|--:|--:|--:|--:|--:|")
for epoch in range(EPOCHS):
    random.shuffle(train)
    start = time.perf_counter()
    for inputs, label in train:
        mlp.learn(mlp.tick(inputs), [float(label == 0), float(label == 1)])
    duration = time.perf_counter() - start
    trainAcc = sum(predicted(mlp, x) == t for x, t in train) / len(train) * 100
    preds = [predicted(mlp, x) for x, _ in test]
    testAcc = sum(p == t for p, (_, t) in zip(preds, test)) / len(test) * 100
    print(f"| {epoch + 1} | {trainAcc:.1f} % | {testAcc:.1f} % | {sum(preds) / len(preds) * 100:.1f} % | {duration:.0f} |", flush=True)
