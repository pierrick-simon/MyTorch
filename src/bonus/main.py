from typeguard import typechecked
from sys import argv
from os import path, walk
from importlib import import_module
import runpy
import sys

SRC_DIR = path.dirname(path.dirname(path.abspath(__file__)))
BONUS_DIR = path.join(SRC_DIR, "bonus")
MODES = {"--learn": "train", "--evaluate": "evaluate"}
HELP_PATH = path.join(BONUS_DIR, "help.txt")

@typechecked
def checkParam() -> int | None:
    if len(argv) != 4 or argv[1] not in MODES:
        print(open(HELP_PATH, "r").read(), end="")
        return 84
    task = argv[2]
    script = path.join(BONUS_DIR, task, MODES[argv[1]])
    if task.startswith(".") or not path.isfile(script):
        print(f"my_torch_bonus: unknown task '{task}'")
        return 84
    if not argv[3].endswith(".nn"):
        print(f"my_torch_bonus: cannot access {argv[3]}: Should be a .nn file")
        return 84
    if argv[1] == "--evaluate" and not path.exists(argv[3]):
        print(f"my_torch_bonus: cannot access {argv[3]}: No such file")
        return 84

@typechecked
def aliasMyTorch() -> None:
    # Networks made by my_torch_generator are pickled as src.my_torch.*, while the
    # bonus scripts import my_torch.*: make both names point to the same modules.
    libDir = path.join(SRC_DIR, "my_torch")
    for root, _, files in walk(libDir):
        for f in files:
            if not f.endswith(".py") or f == "__init__.py":
                continue
            rel = path.relpath(path.join(root, f[:-3]), SRC_DIR)
            import_module("src." + rel.replace(path.sep, "."))
    for name in list(sys.modules):
        if name == "src.my_torch" or name.startswith("src.my_torch."):
            sys.modules[name.removeprefix("src.")] = sys.modules[name]

@typechecked
def dispatch(mode: str, task: str, file: str) -> int:
    taskDir = path.join(BONUS_DIR, task)
    script = path.join(taskDir, MODES[mode])
    aliasMyTorch()
    sys.path.insert(0, taskDir)
    sys.argv = [script, file]
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 84
    return 0

@typechecked
def main() -> int:
    if ret := checkParam():
        return ret
    return dispatch(argv[1], argv[2], argv[3])
