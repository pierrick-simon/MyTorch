from typeguard import typechecked

@typechecked
def AFBool(sum: float, bias: float) -> float:
    return sum >= bias