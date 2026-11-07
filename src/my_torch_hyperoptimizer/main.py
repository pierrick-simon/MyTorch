import random
import sys

from typeguard import typechecked

from src.my_torch_hyperoptimizer.constants import BEST_YML_PATH, NB_TRIALS, RESULTS_PATH, SEED
from src.my_torch_hyperoptimizer.dataset import loadData
from src.my_torch_hyperoptimizer.halving import successiveHalving
from src.my_torch_hyperoptimizer.network import writeYml
from src.my_torch_hyperoptimizer.results import writeResults
from src.my_torch_hyperoptimizer.search_space import sampleConfig
from src.my_torch_hyperoptimizer.trial import Trial

@typechecked
def main() -> int:
    try:
        train, validation = loadData()
    except (ValueError, OSError) as e:
        print("my_torch_hyperoptimizer:", e, file=sys.stderr)
        return 84
    print(f"{len(train)} training positions, {len(validation)} validation positions")

    rng = random.Random(SEED)
    trials = [Trial(sampleConfig(rng), SEED) for _ in range(NB_TRIALS)]
    best, score = successiveHalving(trials, train, validation)
    writeResults(trials, RESULTS_PATH)
    writeYml(best.config, SEED, BEST_YML_PATH)
    print(f"\nBest configuration: {score:.1f} % after {best.seen} positions")
    print(best.config)
    print(f"All the scores are in {RESULTS_PATH}, the best configuration is in {BEST_YML_PATH}")
    return 0
