# ruff: noqa: N999 - module name kept so pickled .nn files still load
import math
import pickle
import random as rand
from os.path import exists

import click
from typeguard import typechecked

from .activationClasses import *
from .perceptron import Perceptron


@typechecked
def initLimit(activationClass: ActivationClass, nbInputs: int, nbOutputs: int) -> float:
    """Weights are drawn in [-limit, limit]: He for ReLU-like functions, Xavier for the others."""
    if activationClass in (ReLU, LeakyReLU):
        return math.sqrt(6 / nbInputs)
    return math.sqrt(6 / (nbInputs + nbOutputs))


class MLP:
    @typechecked
    def __init__(self, nbInputs: int, layers: list[int], nbOutputs: int, learningRate: float, activationClasses: tuple[ActivationClass, ActivationClass | None], seed: float = 67, momentum: float = 0.0, initialBias: float = 0.5, patience: int | None = None, validationInterval: int = 5000, batchSize: int = 1):
        rand.seed(seed)
        self.seed: float = seed
        self.patience: int = patience
        self.validationInterval: int = validationInterval
        self.batchSize: int = batchSize
        self.nbInputs: int = nbInputs
        self.layers: list[list[Perceptron]] = []
        self.hiddenActivationClass: ActivationClass = activationClasses[0]
        self.outputActivationClass: ActivationClass = activationClasses[1]
        for i in range(len(layers)):
            self.layers.append([])
            if i == 0:
                prevLayerSize = nbInputs
            else:
                prevLayerSize = layers[i - 1]
            for _ in range(layers[i]):
                limit = initLimit(self.hiddenActivationClass, prevLayerSize, layers[i])
                weights = [rand.uniform(-limit, limit) for _ in range(prevLayerSize)]
                self.layers[i].append(Perceptron(weights, learningRate, initialBias, momentum, self.hiddenActivationClass.activation))
        self.layers.append([])
        for i in range(nbOutputs):
            limit = initLimit(self.outputActivationClass, len(self.layers[-2]), nbOutputs)
            weights = [rand.uniform(-limit, limit) for _ in range(len(self.layers[-2]))]
            self.layers[-1].append(Perceptron(weights, learningRate, initialBias, momentum, self.outputActivationClass.activation))

    def __setstate__(self, state: dict):
        """Networks saved before mini-batches are trained online."""
        state.setdefault("batchSize", 1)
        self.__dict__.update(state)

    @typechecked
    def __str__(self):
        ret = f"There is {self.nbInputs} inputs layers.\n"
        for i, layer in enumerate(self.layers):
            if i == 0:
                prev = self.nbInputs
            else:
                prev = len(self.layers[i - 1])
            if i != len(self.layers) - 1:
                ret += f"Layer n°{i}. ({prev} inputs, {len(layer)} perceptrons)\n"
            else:
                ret += f"Layer n°{i}. ({prev} inputs, {len(layer)} perceptrons ouputs)"
        return ret

    @typechecked
    def tick(self, inputs: list[float]) -> list[float]:
        for layer in self.layers:
            output = []
            for p in layer:
                output.append(p.tick(inputs))
            inputs = output
        return inputs

    @typechecked
    def errorUnits(self, outputs: list[float], targets: list[float]) -> list[list[float]]:
        """Error of every perceptron for the last sample, from the last layer to the first."""
        errors: list[float] = []
        for output, target in zip(outputs, targets, strict=True):
            outputUnitError = self.outputActivationClass.derivation(output) * (target - output)
            errors.append(outputUnitError)

        perceptrons: list[Perceptron] = []
        errorUnits: list[list[float]] = []
        for i, layer in enumerate(reversed(self.layers)):
            errorUnits.append([])
            lastPerceptrons = perceptrons.copy()
            perceptrons.clear()
            lastErrors = errors.copy()
            errors.clear()
            for j, p in enumerate(layer):
                if i == 0:
                    errorUnit = lastErrors[j]
                else:
                    sum = 0
                    lw = [lp.weights[j] for lp in lastPerceptrons]
                    for e, w in zip(lastErrors, lw, strict=True):
                        sum += e * w
                    errorUnit = self.hiddenActivationClass.derivation(p.lastOutput) * sum
                errorUnits[i].append(errorUnit)

                errors.append(errorUnit)
                perceptrons.append(p)

        return errorUnits

    @typechecked
    def learn(self, outputs: list[float], targets: list[float]):
        """Online training: updates the weights right away with the last sample."""
        errorUnits = self.errorUnits(outputs, targets)
        for i, layer in enumerate(self.layers):
            for j, p in enumerate(layer):
                p.learn(errorUnits[len(errorUnits) - 1 - i][j])

    @typechecked
    def accumulate(self, outputs: list[float], targets: list[float]):
        """Mini-batch training: keeps the update of the last sample until step is called."""
        errorUnits = self.errorUnits(outputs, targets)
        for i, layer in enumerate(self.layers):
            for j, p in enumerate(layer):
                p.accumulate(errorUnits[len(errorUnits) - 1 - i][j])

    @typechecked
    def step(self, batchSize: int):
        """Applies the mean of the updates accumulated over the last batchSize samples."""
        for layer in self.layers:
            for p in layer:
                p.applyAccumulated(batchSize)

    @typechecked
    def save(self, path: str, forceOverwrite: bool = False, printResult: bool = True):
        if not path.endswith(".nn"):
            print("my_torch: Warning: you should save with .nn file extention")
        if forceOverwrite or not exists(path) or click.confirm(f"Do you want to overwrite {path} ?"):
            with open(path, "wb") as f:
                pickle.dump(self, f)
            if printResult:
                print(f"Successfully saved model \'{path}\'.")

    @typechecked
    def load(path: str, printResult: bool = True) -> "MLP":
        if printResult:
            print(f"Successfully loaded model \'{path}\'.")
        with open(path, "rb") as f:
            return pickle.load(f)
