import yaml
from typeguard import typechecked

from src.my_torch.activationClasses import ActivationClassNames

conf = {
    "nbInputs": (int, True),
    "nbOutputs": (int, True),
    "hiderLayersLayout": (list[int], True),
    "learningRate": (float, True),
    "activationFunction": (str, False),
    "hiddenActivationFunction": (str, False),
    "outputActivationFunction": (str, False),
    "initialBias": (float, False),
    "seed": (int, False),
    "patience": (int, False),
}

ConfType = dict[str, int | float | str | list[int]]

@typechecked
def check_label(data: ConfType, label: str, configPath: str):
    if not label in data:
        raise ValueError(f"my_torch generator: YAML config file does not containes the necessary label {label} in {configPath}. Please read the README.md or the exemple .yml config files.")

@typechecked
def check_config(data: ConfType, configPath: str) -> None:
    data_dup = data.copy()
    for key, (value, required) in conf.items():
        if required:
            check_label(data, key, configPath)
        if key in data:
            value(data[key])
            data_dup.pop(key)
    for key in data_dup:
        print(f"my_torch generator: Warning: unessesary labels \'{key}\' in {configPath}")
        data.pop(key)
    default = data.pop("activationFunction", None)
    for key in ("hiddenActivationFunction", "outputActivationFunction"):
        data.setdefault(key, default)
        if data[key] is None:
            raise ValueError(f"my_torch generator: YAML config file needs the label activationFunction, or both hiddenActivationFunction and outputActivationFunction, in {configPath}. Please read the README.md or the exemple .yml config files.")
        if str(data[key]) not in ActivationClassNames:
            raise ValueError(f"my_torch generator: unknown activation function \'{data[key]}\' in {configPath}. Available: {', '.join(ActivationClassNames)}.")
        data[key] = ActivationClassNames[data[key]]
    if data.get("patience", 1) < 1:
        raise ValueError(f"my_torch generator: patience must be at least 1 in {configPath}.")

@typechecked
def parse_yml(configPath: str) -> ConfType:
    data = {}
    with open(configPath, 'r') as stream:
        data = yaml.safe_load(stream)
    check_config(data, configPath)
    return data