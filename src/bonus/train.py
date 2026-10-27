from src.my_torch.MLP import MLP
import random as rand
from typeguard import typechecked

@typechecked
def train(p: MLP, table: list[dict[str, list[float]]], n: int = 100000) -> None:
    for _ in range(n):
        case = rand.randint(0, len(table) - 1)
        output = p.tick(table[case]["inputs"])
        p.learn(output, table[case]["expectedOutputs"])
    print(f"Trained over {n} iterations.")
