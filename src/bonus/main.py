from typeguard import typechecked
from sys import argv
from os import path
from src.my_torch.MLP import MLP
from importlib import import_module

BONUS_DIR = path.dirname(path.abspath(__file__))
HELP_PATH = path.join(BONUS_DIR, "help.txt")
TABLES_DIR = path.join(BONUS_DIR, "tables")
MODES = ["--train", "--eval"]
DEFAULT_ITERATIONS = {"--train": 100000, "--eval": 1000}

@typechecked
def checkParam() -> int | None:
    if len(argv) not in (4, 6) or argv[1] not in MODES:
        print(open(HELP_PATH, "r").read(), end="")
        return 84
    if not argv[2].isidentifier() or not path.isfile(path.join(TABLES_DIR, argv[2] + ".py")):
        print(f"my_torch_bonus: unknown task '{argv[2]}'")
        return 84
    if not argv[3].endswith(".nn"):
        print(f"my_torch_bonus: cannot access {argv[3]}: Should be a .nn file")
        return 84
    if not path.exists(argv[3]):
        print(f"my_torch_bonus: cannot access {argv[3]}: No such file")
        return 84
    if len(argv) == 6 and (argv[4] != "--iteration" or not argv[5].isdigit() or int(argv[5]) == 0):
        print(open(HELP_PATH, "r").read(), end="")
        return 84

@typechecked
def main() -> int:
    if ret := checkParam():
        return ret
    p = MLP.load(argv[3])
    gate = import_module("src.bonus.tables." + argv[2])
    n = int(argv[5]) if len(argv) == 6 else DEFAULT_ITERATIONS[argv[1]]
    if argv[1] == "--train":
        if gate.train(p, n):
            p.save(argv[3], forceOverwrite=True)
    else:
        gate.evaluate(p, n)
    return 0
