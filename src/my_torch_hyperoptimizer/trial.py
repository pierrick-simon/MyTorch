import random

from typeguard import typechecked

from src.my_torch_analyzer.chessboard import Chessboard
from src.my_torch_analyzer.train import accuracy, balancedEpoch, trainBatch
from src.my_torch_hyperoptimizer.network import buildNetwork


class Trial:

    @typechecked
    def __init__(self, config: dict, seed: int):
        self.config = config
        self.mlp = buildNetwork(config, seed)
        self.rng = random.Random(seed)
        self.epoch: list[Chessboard] = []
        self.pos = 0
        self.seen = 0
        self.diverged = False
        self.history: list[tuple[int, float]] = []  # (positions seen, validation accuracy) at each rung

    @typechecked
    def train(self, data: list[Chessboard], budget: int) -> None:
        """Continue the training until the network has seen `budget` positions in total."""
        try:
            while self.seen < budget and not self.diverged:
                if self.pos >= len(self.epoch):
                    self.epoch = balancedEpoch(data, self.rng)
                    self.pos = 0
                size = min(self.mlp.batchSize, budget - self.seen, len(self.epoch) - self.pos)
                trainBatch(self.epoch[self.pos:self.pos + size], self.mlp)
                self.pos += size
                self.seen += size
        except OverflowError:
            self.diverged = True

    @typechecked
    def score(self, validation: list[Chessboard]) -> float:
        return 0.0 if self.diverged else accuracy(validation, self.mlp)
