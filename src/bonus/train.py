import random as rand

import click
from typeguard import typechecked

from src.my_torch.MLP import MLP

ACTIONS = {"s": "save", "r": "resume", "q": "quit"}

@typechecked
def askAfterInterrupt(i: int, n: int) -> str:
    print(f"\nTraining interrupted after {i}/{n} iterations.")
    choices = list(ACTIONS.keys()) + list(ACTIONS.values())
    try:
        answer = click.prompt(f"Save the model, resume training or quit without saving? ({', '.join(ACTIONS.values())})", type=click.Choice(choices, case_sensitive=False), default="save", show_choices=False)
    except click.Abort:
        return "quit"
    return ACTIONS[answer[0].lower()]

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
