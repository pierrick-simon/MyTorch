from typeguard import typechecked

from src.my_torch.MLP import MLP
from src.my_torch_analyzer.Chessboard import Chessboard

@typechecked
def train(data: list[Chessboard], mlp: MLP):
    i = 0
    try:
        for chessboard in data:
            output = mlp.tick(chessboard.getInput())
            mlp.learn(output, chessboard.getSate())
            i += 1
    except KeyboardInterrupt:
        pass
    print(f"Trained over {i} iterations.")