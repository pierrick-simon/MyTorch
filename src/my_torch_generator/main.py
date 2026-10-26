from typeguard import typechecked
from sys import argv
from os import path
from src.my_torch import MLP

HELP_PATH = "src/my_torch_generator/help.txt"

@typechecked
def checkParam() -> int | None:
    if len(argv) < 2 or len(argv) % 2 != 1 or sum([int(not argv[i].isdigit()) for i in range(2, len(argv), 2)]) != 0:
        print(open(HELP_PATH, "r").read())
        return 84
    for i in range(1, len(argv), 2):
        if not argv[i].endswith(".conf"):
            print(f"my_torch_generator: cannot access {argv[i]}: Should be a .yml file")
            return 84
        if not path.exists(argv[i]):
            print(f"my_torch_generator: cannot access {argv[i]}: No such file")
            return 84

@typechecked
def generateNeuralNetwork(confPath: str) -> None:
    print(f"Generated nn with {confPath} config file.")

@typechecked
def generateNeuralNetworks() -> None:
    for i in range(1, len(argv), 2):
        for _ in range(int(argv[i + 1])):
            generateNeuralNetwork(argv[i])

@typechecked
def main() -> int:
    if ret := checkParam():
        return ret
    generateNeuralNetworks()
    return 0