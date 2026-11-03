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
def trainSample(p: MLP, case: dict[str, list[float]], i: int):
    """Trains on sample number i, updating the weights at the end of each batch."""
    output = p.tick(case["inputs"])
    if p.batchSize == 1:
        p.learn(output, case["expectedOutputs"])
        return
    p.accumulate(output, case["expectedOutputs"])
    if (i + 1) % p.batchSize == 0:
        p.step(p.batchSize)

@typechecked
def flushBatch(p: MLP, i: int):
    """Applies the samples of an unfinished batch."""
    if p.batchSize > 1 and i % p.batchSize:
        p.step(i % p.batchSize)

@typechecked
def train(p: MLP, table: list[dict[str, list[float]]], n: int = 100000) -> bool:
    i = 0
    while i < n:
        try:
            while i < n:
                trainSample(p, table[rand.randint(0, len(table) - 1)], i)
                i += 1
        except KeyboardInterrupt:
            action = askAfterInterrupt(i, n)
            if action == "resume":
                continue
            flushBatch(p, i)
            print(f"Trained over {i} iterations.")
            return action == "save"
    flushBatch(p, n)
    print(f"Trained over {n} iterations.")
    return True
