from typeguard import typechecked
import yaml

conf = {
    "nbInputs": int,
    "nbOutputs": int,
    "hiderLayersLayout": list[int],
    "learningRate": float,
}

conftype = dict[str, int | float | list]

@typechecked
def check_label(data: conftype, label: str, configPath: str):
    if not label in data:
        raise ValueError(f"my_torch generator: YAML config file does not containes the necessary label {label} in {configPath}. Please read the README.md or the exemple .yml config files.")

@typechecked
def check_config(data: conftype, configPath: str) -> None:
    data_dup = data.copy()
    for key, value in conf.items():
        check_label(data, key, configPath)
        value(data[key])
        data_dup.pop(key)
    for key, _ in data_dup.items():
        print(f"my_torch generator: Warning: unessesary labels \'{key}\' in {configPath}")
        data.pop(key)

@typechecked
def parse_yml(configPath: str) -> conftype:
    data = {}
    with open(configPath, 'r') as stream:
        data = yaml.safe_load(stream)
    check_config(data, configPath)
    return data