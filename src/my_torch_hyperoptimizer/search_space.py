import math
import random

from typeguard import typechecked

# kind, then its arguments:
# - ("choice", values): one of the values
# - ("uniform", low, high): a float between low and high
# - ("loguniform", low, high): a float between low and high, drawn on a log scale (for the learning rate)
SEARCH_SPACE = {
    "hiderLayersLayout": ("choice", [[32], [64], [128], [64, 32], [128, 64], [128, 64, 32], [256, 64]]),
    "learningRate": ("loguniform", 1e-3, 5e-2),
    "momentum": ("uniform", 0.0, 0.95),
    "hiddenActivationFunction": ("choice", ["ReLU", "LeakyReLU", "Tanh", "Sigmoid"]),
    "outputActivationFunction": ("choice", ["Sigmoid"]),
    "initialBias": ("uniform", 0.0, 0.5),
    "batchSize": ("choice", [1, 4, 16]),
}

@typechecked
def sampleValue(spec: tuple, rng: random.Random) -> int | float | str | list[int]:
    kind = spec[0]
    if kind == "choice":
        return rng.choice(spec[1])
    if kind == "uniform":
        return round(rng.uniform(spec[1], spec[2]), 4)
    if kind == "loguniform":
        return float(f"{10 ** rng.uniform(math.log10(spec[1]), math.log10(spec[2])):.3g}")
    raise ValueError(f"unknown search space kind {kind!r}")

@typechecked
def sampleConfig(rng: random.Random) -> dict:
    return {name: sampleValue(spec, rng) for name, spec in SEARCH_SPACE.items()}
