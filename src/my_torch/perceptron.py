from collections.abc import Callable

from typeguard import typechecked


class Perceptron:
    @typechecked
    def __init__(self, weights: list[float], learningRate: float, bias: float, momentum: float, activationFunc: Callable):
        self.weights: list[float] = weights
        self.bias: float = bias
        self.momentum: float = momentum
        self.lr: float = learningRate
        self.lastInput: list[float] = []
        self.activationFunc: Callable = activationFunc
        self.lastDelta: list[float] = [0.0 for _ in range(len(self.weights))]

    @typechecked
    def __str__(self) -> str:
        return f"weights: {self.weights}\nbias: {self.bias}\n"
    
    @typechecked
    def tick(self, inputs: list[float]) -> float:
        if (len(inputs) != len(self.weights)):
            raise ValueError("lenght of inputs is not the same as lenght of weights")
        self.lastInput = inputs
        sum: float = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        self.lastOutput = self.activationFunc(sum, self.bias)
        return self.lastOutput

    @typechecked
    def learn(self, unitError: float):
        for i in range(len(self.weights)):
            delta: float = (self.lr * unitError * self.lastInput[i]) + (self.momentum * self.lastDelta[i])
            self.lastDelta[i] = delta
            self.weights[i] += delta
        self.bias += self.lr * unitError
