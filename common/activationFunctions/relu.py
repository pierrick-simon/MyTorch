from typeguard import typechecked

@typechecked
def AFRelu(sum: float, bias: float) -> float:
    return max(0, sum + bias)