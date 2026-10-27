from .perceptron import Perceptron
import random as rand
import pickle
from collections.abc import Callable
from typeguard import typechecked
import click

class MLP:
    @typechecked
    def __init__(self, nbInputs: int, layers: list[int], nbOutputs: int, learningRate: float, activationFunc: Callable, seed: int | float = 67):
        rand.seed(seed)
        self.seed = seed
        self.nbInputs = nbInputs
        self.layers = []
        for i in range(len(layers)):
            self.layers.append([])
            if i == 0:
                prevLayerSize = nbInputs
            else:
                prevLayerSize = layers[i - 1]
            for _ in range(layers[i]):
                weights = [rand.uniform(-1, 1) for _ in range(prevLayerSize)]
                self.layers[i].append(Perceptron(weights, learningRate, 0.5, activationFunc))
        self.layers.append([])
        for i in range(nbOutputs):
            weights = [rand.uniform(-1, 1) for _ in range(len(self.layers[-2]))]
            self.layers[-1].append(Perceptron(weights, learningRate, 0.5, activationFunc))

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
    def learn(self, outputs: list[float], targets: list[float]):
        errors: list[float] = []
        for output, target in zip(outputs, targets, strict=True):
            outputUnitError = output * (1 - output) * (target - output)
            errors.append(outputUnitError)

        perceptrons = []
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
                    errorUnit = p.lastOutput * (1 - p.lastOutput) * sum
                errorUnits[i].append(errorUnit)

                errors.append(errorUnit)
                perceptrons.append(p)

        for i, layer in enumerate(self.layers):
            for j, p in enumerate(layer):
                p.learn(errorUnits[len(errorUnits) - 1 - i][j])

    @typechecked
    def save(self, path: str, forceOverwrite: bool = False):
        if not path.endswith(".nn"):
            print("my_torch: Warning: you should save with .nn file extention")
        if forceOverwrite or click.confirm(f"Do you want to overwrite {path} ?"):
            pickle.dump(self, open(path, "wb"))
            print(f"Successfully saved model \'{path}\'.")

    @typechecked
    def load(path: str) -> MLP:
        print(f"Successfully loaded model \'{path}\'.")
        return pickle.load(open(path, "rb"))
