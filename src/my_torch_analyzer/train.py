import os
import random
import sys

import click
from tqdm import tqdm
from typeguard import typechecked

from src.my_torch.MLP import MLP
from src.my_torch_analyzer.chessboard import Chessboard

EPOCHS = 20
CHECKPOINT_EVERY = 1000
VALIDATION_SIZE = 500  # at most, and at most 10 % of the data
MAX_OVERSAMPLING = 4

@typechecked
def labelIndex(chessboard: Chessboard) -> int:
    state = chessboard.getSate()
    return state.index(max(state))


@typechecked
def splitValidation(data: list[Chessboard], seed: float) -> tuple[list[Chessboard], list[Chessboard]]:
    size = min(VALIDATION_SIZE, len(data) // 10)
    indexes = set(random.Random(seed).sample(range(len(data)), size))
    trainData = [c for i, c in enumerate(data) if i not in indexes]
    validationData = [c for i, c in enumerate(data) if i in indexes]
    return trainData, validationData


@typechecked
def balancedEpoch(data: list[Chessboard], rng: random.Random) -> list[Chessboard]:
    """One shuffled epoch where rare classes are oversampled (with replacement)."""
    byClass: dict[int, list[Chessboard]] = {}
    for chessboard in data:
        byClass.setdefault(labelIndex(chessboard), []).append(chessboard)
    if not byClass:
        return []
    target = max(len(items) for items in byClass.values())
    epoch: list[Chessboard] = []
    for items in byClass.values():
        extra = min(target, MAX_OVERSAMPLING * len(items)) - len(items)
        epoch.extend(items)
        epoch.extend(rng.choices(items, k=extra))
    rng.shuffle(epoch)
    return epoch

@typechecked
def accuracy(data: list[Chessboard], mlp: MLP) -> float:
    ok = 0
    for chessboard in data:
        output = mlp.tick(chessboard.getInput())
        ok += output.index(max(output)) == labelIndex(chessboard)
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
    tqdm.write(f"  validation {current:.1f} %")
    if state["worse"] < mlp.patience:
        return False
    state["worse"] = 0
    return proposeStop(pbar, state["best"], state["iteration"], mlp.patience)

@typechecked
def prepareSaveDir(savePath: str) -> str:
    """Create a clean output directory, refusing to wipe anything that is not ours."""
    savePath = savePath.removesuffix(".nn")
    if os.path.isdir(savePath):
        files = os.listdir(savePath)
        if any(not name.endswith(".nn") for name in files):
            raise ValueError(f"refusing to overwrite {savePath!r}: it contains files that are not .nn models")
        for name in files:
            os.remove(os.path.join(savePath, name))
    elif os.path.exists(savePath):
        raise ValueError(f"{savePath!r} exists and is not a directory")
    else:
        os.makedirs(savePath)
    return savePath

@typechecked
def train(data: list[Chessboard], mlp: MLP, savePath: str = "train") -> MLP:
    savePath = prepareSaveDir(savePath)
    bestPath = f"{savePath}/best.nn"

    rng = random.Random(mlp.seed)
    data = list(data)
    rng.shuffle(data)
    validationData: list[Chessboard] = []
    if mlp.patience is not None:
        data, validationData = splitValidation(data, mlp.seed)

    state = None
    if validationData:
        state = {"best": accuracy(validationData, mlp), "iteration": 0, "worse": 0}
        print(f"{len(validationData)} positions kept aside for validation, accuracy before training: {state['best']:.1f} %")
        mlp.save(bestPath, True, False)

    i = 0
    try:
        for epoch in range(EPOCHS):
            epochData = balancedEpoch(data, rng)
            pbar = tqdm(epochData, desc=f"epoch {epoch + 1}/{EPOCHS}")
            try:
                for chessboard in pbar:
                    if i % CHECKPOINT_EVERY == 0:
                        mlp.save(f"{savePath}/model_{i}.nn", True, False)
                    output = mlp.tick(chessboard.getInput())
                    mlp.learn(output, chessboard.getSate())
                    i += 1
                stop = bool(validationData) and checkPatience(validationData, mlp, i, pbar, bestPath, state)
            finally:
                pbar.close()
            if stop:
                print(f"Stopped after {i} iterations, keeping the model of iteration {state['iteration']}.")
                return MLP.load(bestPath, False)
    except KeyboardInterrupt:
        pass

    print(f"Trained over {i} iterations.")
    if validationData:
        last = accuracy(validationData, mlp)
        print(f"Validation accuracy: {last:.1f} % (best: {state['best']:.1f} % at iteration {state['iteration']}, saved in {bestPath})")
        if state["best"] > last:
            print("Returning the best model instead of the last one.")
            return MLP.load(bestPath, False)
    return mlp