from typeguard import typechecked
from sys import argv
from os import path
from src.my_torch.MLP import MLP
from src.my_torch.activationClass.sigmoid import Sigmoid
from src.my_torch_generator.parse_yml import parse_yml, conftype
from datetime import datetime

HELP_PATH = "src/my_torch_generator/help.txt"

@typechecked
def checkParam() -> int | None:
    if len(argv) < 2 or len(argv) % 2 != 1 or sum([int(not argv[i].isdigit()) for i in range(2, len(argv), 2)]) != 0:
        print(open(HELP_PATH, "r").read())
        return 84
    for i in range(1, len(argv), 2):
        if not argv[i].endswith(".yml"):
            print(f"my_torch_generator: cannot access {argv[i]}: Should be a .yml file")
            return 84
        if not path.exists(argv[i]):
            print(f"my_torch_generator: cannot access {argv[i]}: No such file")
            return 84

@typechecked
def generateNeuralNetwork(data: conftype, path: str) -> None:
    MLP(data["nbInputs"], data["hiderLayersLayout"], data["nbOutputs"], data["learningRate"], Sigmoid, datetime.now().timestamp()).save(path)

@typechecked
def generateNeuralNetworks() -> None:
    for i in range(1, len(argv), 2):
        data = parse_yml(argv[i])
        for j in range(int(argv[i + 1])):
            generateNeuralNetwork(data, argv[i].replace(".yml", "") + f"_{j + 1}.nn")

@typechecked
def main() -> int:
    if ret := checkParam():
        return ret
    generateNeuralNetworks()
    return 0