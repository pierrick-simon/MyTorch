from src.my_torch.MLP import MLP
from src.bonus.train import train as genericTrain
from src.bonus.evaluate import evaluate as genericEvaluate
from itertools import product
from typeguard import typechecked

TABLE = [
    {"inputs": list(bits), "expectedOutputs": [sum(a != b for a, b in zip(bits[:3], bits[3:]))]}
    for bits in product([0, 1], repeat=6)
]

@typechecked
def train(p: MLP, n: int) -> bool:
    return genericTrain(p, TABLE, n)

@typechecked
def evaluate(p: MLP, n: int) -> None:
    genericEvaluate(p, TABLE, n, threshold=0.5)
