import argparse
import sys

from typeguard import typechecked

from src.my_torch_analyzer.fen_parser import fen_parser
from src.my_torch.MLP import MLP
from src.my_torch_analyzer.train import train
from src.my_torch_analyzer.predict import predict

HELP_PATH = "src/my_torch_analyzer/help.txt"

class ArgParser(argparse.ArgumentParser):
    def error(self, message: str):
        print(f"my_torch_analyzer: {message}", file=sys.stderr)
        sys.exit(84)

@typechecked
def build_parser() -> ArgParser:
    p = ArgParser(add_help=False)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predict", action="store_true")
    mode.add_argument("--train", action="store_true")
    p.add_argument("--save", metavar="SAVEFILE", type=str, default=None)
    p.add_argument("LOADFILE", type=str)
    p.add_argument("CHESSFILE", type=str)
    return p


@typechecked
def main() -> int:
    argv = sys.argv[1:]

    if "--help" in argv or "-h" in argv:
        print(open(HELP_PATH, "r").read())
        return 0

    args = build_parser().parse_args(argv)

    if args.save is not None and not args.train:
        print("my_torch_analyzer: --save only works in train mode", file=sys.stderr)
        return 84

    try:
        data = fen_parser(args.CHESSFILE)
        mlp: MLP = MLP.load(args.LOADFILE)
        if args.train:
            if any(not chessboard.hasState() for chessboard in data):
                raise ValueError("training mode requires an expected output on every line")
            print("Start Training")
            save_path = args.save or args.LOADFILE
            train(data, mlp)
            mlp.save(save_path, True)
        else:
            predict(data, mlp)
    except (ValueError, OSError) as e:
        print("my_torch_analyzer:", e, file=sys.stderr)
        return 84

    return 0
