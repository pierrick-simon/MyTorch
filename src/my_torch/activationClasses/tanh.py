import math

from typeguard import typechecked


class Tanh:
    @typechecked
    def activation(sum: float, bias: float) -> float:
        return math.tanh(sum + bias)

    @typechecked
    def derivation(x: float) -> float:
        return 1 - x ** 2
