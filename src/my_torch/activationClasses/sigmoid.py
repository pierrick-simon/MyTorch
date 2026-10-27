import math

from typeguard import typechecked


class Sigmoid:
    @typechecked
    def activation(sum: float, bias: float) -> float:
        return 1 / (1 + math.exp(-sum - bias))

    @typechecked
    def derivation(x: float) -> float:
        return x * (1 - x)