"""Time of one forward pass and one training step for several network shapes."""
import sys
import time
from os import path

sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))
from src.my_torch.activationClasses import ReLU, Sigmoid
from src.my_torch.MLP import MLP

SHAPES = [
    ("xor", 2, [2], 1, ReLU),
    ("hamming", 6, [8], 1, ReLU),
    ("chess small", 783, [32], 2, ReLU),
    ("chess medium", 783, [64, 32], 2, ReLU),
    ("chess large", 783, [256, 64, 32], 3, Sigmoid),  # ReLU overflows the Sigmoid output with this shape
]

print("| Network | Shape | Parameters | Forward (ms) | Training step (ms) | Steps / s |")
print("|---|---|--:|--:|--:|--:|")
for name, nbInputs, layers, nbOutputs, hidden in SHAPES:
    mlp = MLP(nbInputs, layers, nbOutputs, 0.01, (hidden, Sigmoid), 1)
    inputs = [float(i % 2) for i in range(nbInputs)]
    target = [1.0] + [0.0] * (nbOutputs - 1)
    n = 2000 if nbInputs < 10 else 20
    start = time.perf_counter()
    for _ in range(n):
        mlp.tick(inputs)
    forward = (time.perf_counter() - start) / n
    start = time.perf_counter()
    for _ in range(n):
        mlp.learn(mlp.tick(inputs), target)
    step = (time.perf_counter() - start) / n
    params = sum(len(p.weights) + 1 for layer in mlp.layers for p in layer)
    shape = "-".join(map(str, [nbInputs, *layers, nbOutputs]))
    print(f"| {name} | {shape} | {params} | {forward * 1000:.3f} | {step * 1000:.3f} | {1 / step:.0f} |")
