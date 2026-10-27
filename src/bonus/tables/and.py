from src.my_torch.MLP import MLP
from src.bonus.train import train as genericTrain
from src.bonus.evaluate import evaluate as genericEvaluate
from typeguard import typechecked

TABLE = [
    {"inputs": [1, 1], "expectedOutputs": [1]},
    {"inputs": [1, 0], "expectedOutputs": [0]},
    {"inputs": [0, 1], "expectedOutputs": [0]},
    {"inputs": [0, 0], "expectedOutputs": [0]},
]

@typechecked
def train(p: MLP, n: int) -> bool:
    return genericTrain(p, TABLE, n)

@typechecked
def evaluate(p: MLP, n: int) -> None:
    genericEvaluate(p, TABLE, n)
