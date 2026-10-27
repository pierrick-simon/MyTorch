from typeguard import typechecked

from src.my_torch.MLP import MLP
from src.my_torch_analyzer.Chessboard import Chessboard, getStateStr

@typechecked
def predict(data: list[Chessboard], mlp: MLP):
    try:
        for chessboard in data:
            output = mlp.tick(chessboard.getInput())
            print(output)
            print(getStateStr(output))
    except KeyboardInterrupt:
        pass
