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
        self.deltaSum: list[float] = [0.0 for _ in range(len(self.weights))]
        self.biasDeltaSum: float = 0.0

    def __setstate__(self, state: dict):
        """Networks saved before mini-batches have no update sums."""
        state.setdefault("deltaSum", [0.0 for _ in range(len(state["weights"]))])
        state.setdefault("biasDeltaSum", 0.0)
        self.__dict__.update(state)

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

    @typechecked
    def accumulate(self, unitError: float):
        """Adds the update of the last sample to the sums, without changing the weights."""
        for i in range(len(self.weights)):
            self.deltaSum[i] += self.lr * unitError * self.lastInput[i]
        self.biasDeltaSum += self.lr * unitError

    @typechecked
    def applyAccumulated(self, batchSize: int):
        """Applies the mean of the accumulated updates, then resets the sums."""
        for i in range(len(self.weights)):
            delta: float = self.deltaSum[i] / batchSize + (self.momentum * self.lastDelta[i])
            self.lastDelta[i] = delta
            self.weights[i] += delta
            self.deltaSum[i] = 0.0
        self.bias += self.biasDeltaSum / batchSize
        self.biasDeltaSum = 0.0
