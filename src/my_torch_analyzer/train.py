import os
import random
import shutil
import sys

import click
from tqdm import tqdm
from typeguard import typechecked

from src.my_torch.MLP import MLP
from src.my_torch_analyzer.chessboard import Chessboard

CHECKPOINT_EVERY = 1000
VALIDATION_SIZE = 500  # at most, and at most 10 % of the data


@typechecked
def splitValidation(data: list[Chessboard], seed: float) -> tuple[list[Chessboard], list[Chessboard]]:
    size = min(VALIDATION_SIZE, len(data) // 10)
    indexes = set(random.Random(seed).sample(range(len(data)), size))
    trainData = [c for i, c in enumerate(data) if i not in indexes]
    validationData = [c for i, c in enumerate(data) if i in indexes]
    return trainData, validationData


@typechecked
def accuracy(data: list[Chessboard], mlp: MLP) -> float:
    ok = 0
    for chessboard in data:
        output = mlp.tick(chessboard.getInput())
        state = chessboard.getSate()
        ok += output.index(max(output)) == state.index(max(state))
    return ok / len(data) * 100


@typechecked
def proposeStop(pbar: tqdm, best: float, bestIteration: int, patience: int) -> bool:
    message = f"Validation accuracy has stayed below its best ({best:.1f} % at iteration {bestIteration}) for {patience} validations in a row."
    if not sys.stdin.isatty():
        tqdm.write(f"Warning: {message}")
        return False
    pbar.clear()
    return click.confirm(f"{message}\nStop training and keep the best model?", default=True)


@typechecked
def checkPatience(data: list[Chessboard], mlp: MLP, iteration: int, pbar: tqdm, bestPath: str, state: dict) -> bool:
    current = accuracy(data, mlp)
    if current > state["best"]:
        state.update(best=current, iteration=iteration, worse=0)
        mlp.save(bestPath, True, False)
    else:
        state["worse"] += 1
    pbar.set_postfix(validation=f"{current:.1f} %", best=f"{state['best']:.1f} %")
    if state["worse"] < mlp.patience:
        return False
    state["worse"] = 0
    return proposeStop(pbar, state["best"], state["iteration"], mlp.patience)


@typechecked
def train(data: list[Chessboard], mlp: MLP, savePath: str = "train") -> MLP:
    savePath = savePath.replace(".nn", "")
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    os.makedirs(savePath)
    bestPath = f"{savePath}/best.nn"
    validationData = []
    if mlp.patience is not None:
        data, validationData = splitValidation(data, mlp.seed)
    if validationData:
        state = {"best": accuracy(validationData, mlp), "iteration": 0, "worse": 0}
        print(f"{len(validationData)} positions kept aside for validation, accuracy before training: {state['best']:.1f} %")
        mlp.save(bestPath, True, False)
    i = 0
    pbar = tqdm(data)
    try:
        for chessboard in pbar:
            if i % CHECKPOINT_EVERY == 0:
                mlp.save(f"{savePath}/model_{i}.nn", True, False)
            output = mlp.tick(chessboard.getInput())
            mlp.learn(output, chessboard.getSate())
            i += 1
            if validationData and i % mlp.validationInterval == 0 and checkPatience(validationData, mlp, i, pbar, bestPath, state):
                pbar.close()
                print(f"Stopped after {i} iterations, keeping the model of iteration {state['iteration']}.")
                return MLP.load(bestPath, False)
    except KeyboardInterrupt:
        pass
    pbar.close()
    print(f"Trained over {i} iterations.")
    if validationData:
        print(f"Validation accuracy: {accuracy(validationData, mlp):.1f} % (best: {state['best']:.1f} % at iteration {state['iteration']}, saved in {bestPath})")
    return mlp
