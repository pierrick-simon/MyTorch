from typeguard import typechecked
import math

@typechecked
def AFSigmoid(sum: float, bias: float) -> float:
    return 1 / (1 + math.exp(-sum - bias))