from typeguard import typechecked


class LeakyReLU:
    @typechecked
    def activation(sum: float, bias: float) -> float:
        return max(0.01 * (sum + bias), sum + bias)

    @typechecked
    def derivation(x: float) -> float:
        return 1 if x > 0 else 0.01
