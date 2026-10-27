from typeguard import typechecked

class Linear:
    @typechecked
    def activation(sum: float, bias: float) -> float:
        return sum + bias

    @typechecked
    def derivation(x: float) -> float:
        return 1
