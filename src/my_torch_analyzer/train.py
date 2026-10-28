import os
import shutil

from tqdm import tqdm
from typeguard import typechecked

from src.my_torch.MLP import MLP
from src.my_torch_analyzer.chessboard import Chessboard


@typechecked
def train(data: list[Chessboard], mlp: MLP, savePath: str = "train"):
    savePath = savePath.replace(".nn", "")
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    os.makedirs(savePath)
    i = 0
    try:
        for chessboard in tqdm(data):
            if i % 1000 == 0:
                mlp.save(f"{savePath}/model_{i}.nn", True, False)
            output = mlp.tick(chessboard.getInput())
            mlp.learn(output, chessboard.getSate())
            i += 1
    except KeyboardInterrupt:
        pass
    print(f"Trained over {i} iterations.")