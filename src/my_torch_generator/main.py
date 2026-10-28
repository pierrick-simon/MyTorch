from datetime import datetime
from os import path
from sys import argv

from typeguard import typechecked

from src.my_torch.MLP import MLP
from src.my_torch_generator.parse_yml import ConfType, parse_yml

HELP_PATH = "src/my_torch_generator/help.txt"

@typechecked
def checkParam() -> int | None:
    if len(argv) < 2 or len(argv) % 2 != 1 or sum([int(not argv[i].isdigit()) for i in range(2, len(argv), 2)]) != 0:
        with open(HELP_PATH) as f:
            print(f.read())
        return 84
    for i in range(1, len(argv), 2):
        if not argv[i].endswith(".yml"):
            print(f"my_torch_generator: cannot access {argv[i]}: Should be a .yml file")
            return 84
        if not path.exists(argv[i]):
            print(f"my_torch_generator: cannot access {argv[i]}: No such file")
            return 84

@typechecked
def generateNeuralNetwork(data: ConfType, path: str, index: int) -> None:
    activations = (data["hiddenActivationFunction"], data["outputActivationFunction"])
    seed = data["seed"] + index if "seed" in data else datetime.now().timestamp()
    MLP(data["nbInputs"], data["hiderLayersLayout"], data["nbOutputs"], data["learningRate"], activations, seed, data.get("initialBias", 0.5)).save(path)

@typechecked
def generateNeuralNetworks() -> None:
    for i in range(1, len(argv), 2):
        data = parse_yml(argv[i])
        for j in range(int(argv[i + 1])):
            generateNeuralNetwork(data, argv[i].replace(".yml", "") + f"_{j + 1}.nn", j)

@typechecked
def main() -> int:
    if ret := checkParam():
        return ret
    generateNeuralNetworks()
    return 0