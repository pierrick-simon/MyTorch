from typeguard import typechecked

from src.my_torch_analyzer.chessboard import Chessboard
from src.my_torch_hyperoptimizer.constants import ETA, MIN_BUDGET
from src.my_torch_hyperoptimizer.trial import Trial


@typechecked
def successiveHalving(trials: list[Trial], train: list[Chessboard], validation: list[Chessboard]) -> tuple[Trial, float]:
    """Train every trial on the budget, keep the best 1/ETA, multiply the budget by ETA.
    Once ETA trials or fewer are left, return the best one and its score."""
    budget = MIN_BUDGET
    while True:
        print(f"\n{len(trials)} configurations, {budget} positions each")
        ranked = []
        for trial in trials:
            trial.train(train, budget)
            score = trial.score(validation)
            trial.history.append((trial.seen, score))
            ranked.append((score, trial))
            print(f"  {score:5.1f} %  {trial.config}")
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        if len(ranked) <= ETA:
            return ranked[0][1], ranked[0][0]
        trials = [trial for _, trial in ranked[:len(ranked) // ETA]]
        budget *= ETA
