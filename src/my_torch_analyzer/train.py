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
VALIDATION_EVERY = 5000
VALIDATION_SIZE = 500  # at most, and at most 10 % of the data
PATIENCE = 5  # validations in a row below the best one before proposing to stop


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
def proposeStop(pbar: tqdm, best: float, bestIteration: int) -> bool:
    message = f"Validation accuracy has stayed below its best ({best:.1f} % at iteration {bestIteration}) for {PATIENCE} validations in a row."
    if not sys.stdin.isatty():
        tqdm.write(f"Warning: {message}")
        return False
    pbar.clear()
    return click.confirm(f"{message}\nStop training and keep the best model?", default=True)


@typechecked
def train(data: list[Chessboard], mlp: MLP, savePath: str = "train") -> MLP:
    savePath = savePath.replace(".nn", "")
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    os.makedirs(savePath)
    bestPath = f"{savePath}/best.nn"
    data, validationData = splitValidation(data, mlp.seed)
    best = accuracy(validationData, mlp) if validationData else 0.0
    bestIteration = 0
    worse = 0
    if validationData:
        print(f"{len(validationData)} positions kept aside for validation, accuracy before training: {best:.1f} %")
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
            if validationData and i % VALIDATION_EVERY == 0:
                current = accuracy(validationData, mlp)
                if current > best:
                    best, bestIteration, worse = current, i, 0
                    mlp.save(bestPath, True, False)
                else:
                    worse += 1
                pbar.set_postfix(validation=f"{current:.1f} %", best=f"{best:.1f} %")
                if worse < PATIENCE:
                    continue
                if proposeStop(pbar, best, bestIteration):
                    pbar.close()
                    print(f"Stopped after {i} iterations, keeping the model of iteration {bestIteration}.")
                    return MLP.load(bestPath, False)
                worse = 0
    except KeyboardInterrupt:
        pass
    pbar.close()
    print(f"Trained over {i} iterations.")
    if validationData:
        print(f"Validation accuracy: {accuracy(validationData, mlp):.1f} % (best: {best:.1f} % at iteration {bestIteration}, saved in {bestPath})")
    return mlp
