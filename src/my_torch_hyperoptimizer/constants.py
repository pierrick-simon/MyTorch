from src.my_torch_analyzer import chessboard

NB_INPUTS = chessboard.NB_INPUTS
NB_OUTPUTS = len(chessboard.STATE_LABELS)

SEED = 42

PATIENCE = 5
VALIDATION_INTERVAL = 5000

PER_FILE = 1000
DATASET_PATH = f"hyperopt/dataset_{PER_FILE}.txt"
VALIDATION_RATIO = 0.2

NB_TRIALS = 27
ETA = 3
MIN_BUDGET = 5000
RESULTS_PATH = "hyperopt/results.csv"
BEST_YML_PATH = "hyperopt/best.yml"
