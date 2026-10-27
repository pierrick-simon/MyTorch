from src.my_torch.MLP import MLP
import random as rand
from typeguard import typechecked
import click

ACTIONS = ["save", "resume", "quit"]

@typechecked
def askAfterInterrupt(i: int, n: int) -> str:
    print(f"\nTraining interrupted after {i}/{n} iterations.")
    try:
        return click.prompt("Save the model, resume training or quit without saving?", type=click.Choice(ACTIONS), default="save")
    except click.Abort:
        return "quit"

@typechecked
def train(p: MLP, table: list[dict[str, list[float]]], n: int = 100000) -> bool:
    i = 0
    while i < n:
        try:
            while i < n:
                case = rand.randint(0, len(table) - 1)
                output = p.tick(table[case]["inputs"])
                p.learn(output, table[case]["expectedOutputs"])
                i += 1
        except KeyboardInterrupt:
            action = askAfterInterrupt(i, n)
            if action == "resume":
                continue
            print(f"Trained over {i} iterations.")
            return action == "save"
    print(f"Trained over {n} iterations.")
    return True
