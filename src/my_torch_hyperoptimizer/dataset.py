import os
import subprocess

from typeguard import typechecked

from src.my_torch_analyzer.chessboard import Chessboard
from src.my_torch_analyzer.fen_parser import fen_parser
from src.my_torch_hyperoptimizer.constants import (
    DATASET_PATH,
    PER_FILE,
    VALIDATION_RATIO,
)


@typechecked
def loadData() -> tuple[list[Chessboard], list[Chessboard]]:
    if not os.path.exists(DATASET_PATH):
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        if subprocess.run(["bash", "dataset.sh", str(PER_FILE), DATASET_PATH], check=False).returncode != 0:
            raise ValueError("dataset.sh failed")
    data = fen_parser(DATASET_PATH)
    size = int(len(data) * VALIDATION_RATIO)
    return data[size:], data[:size]
