import argparse

from src.my_torch_analyzer.fen_parser import fen_parser


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("CHESSFILE", type=str, help="File containing chessboards")
    args = p.parse_args()
    try:
        fen_parser(args.CHESSFILE)
    except ValueError as e:
        print("my_torch_analyser: ", e)
        return 84
    return 0