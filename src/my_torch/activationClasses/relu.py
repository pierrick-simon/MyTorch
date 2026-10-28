from typeguard import typechecked


class ReLU:
    @typechecked
    def activation(sum: float, bias: float) -> float:
        return max(0, sum + bias)

    @typechecked
    def derivation(x: float) -> float:
        return 1 if x > 0 else 0
