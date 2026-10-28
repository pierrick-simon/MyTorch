import os
import shutil

from typeguard import typechecked

from src.my_torch.MLP import MLP
from src.my_torch_analyzer.Chessboard import Chessboard


@typechecked
def train(data: list[Chessboard], mlp: MLP):
    if os.path.exists("train"):
        shutil.rmtree("train")
    os.makedirs("train")
    i = 0
    try:
        for chessboard in data:
            if i % 100 == 0:
                print(f"Iteration {i}/{data.__len__()}. {i / data.__len__() * 100}%")
            if i % 1000 == 0:
                mlp.save(f"train/model_{i}.nn", True)
            output = mlp.tick(chessboard.getInput())
            mlp.learn(output, chessboard.getSate())
            i += 1
    except KeyboardInterrupt:
        pass
    print(f"Trained over {i} iterations.")