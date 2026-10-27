import random as rand

from typeguard import typechecked

from src.my_torch.MLP import MLP


@typechecked
def evaluate(p: MLP, table: list[dict[str, list[float]]], n: int = 1000, threshold: float = 0.1) -> None:
    rand.seed(p.seed ** 2)
    accuracy = 0
    nbOutput = len(table[0]["expectedOutputs"])
    for _ in range(n):
        case = rand.randint(0, len(table) - 1)
        output = p.tick(table[case]["inputs"])
        for o, e in zip(output, table[case]["expectedOutputs"], strict=True):
            accuracy += (abs(o - e) < threshold) / (n * nbOutput)
    print(f"Accuracy is {accuracy * 100.0:.1f}% over {n} iterations.")
