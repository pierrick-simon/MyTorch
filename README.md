# MyTorch

### Development

By *Ariel Amriou*(ariel.amriou@epitech.eu) & *Pierrick Simon*(pierrick.simon@epitech.eu).

MyTorch is a neural network library written from scratch in pure Python (multi-layer perceptrons trained by backpropagation), with four programs built on it:

| Program              | Role                                                                            |
|----------------------|---------------------------------------------------------------------------------|
| `my_torch_generator` | Creates untrained networks (`.nn` files) from YAML configuration files.         |
| `my_torch_analyzer`  | Trains a network on chess positions, or uses it to tell whether a king is in check or checkmate, and by which side. |
| `my_torch_bonus`     | Trains and evaluates a network on small truth tables (AND, XOR, Hamming distance). |
| `my_torch_hyperoptimizer` | Searches the best hyperparameters for the chess analyzer (random search + successive halving). |

### Documentation

| Document                                        | Content                                                                 |
|-------------------------------------------------|-------------------------------------------------------------------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)     | Code layout, forward pass, backpropagation, `.nn` file format, how each program works. |
| [docs/CHESS_ANALYZER.md](docs/CHESS_ANALYZER.md) | Chess file format, input and output encoding, dataset, training, checkpoints and validation. |
| [docs/DESIGN_CHOICES.md](docs/DESIGN_CHOICES.md) | Why the project is built this way, and what each choice costs.          |
| [docs/CHOICE_JUSTIFICATION.md](docs/CHOICE_JUSTIFICATION.md) | Why each major decision was made, its alternatives and the evidence, and how the choices answer the constraints of the project. |
| [docs/ALTERNATIVE_PROPOSAL.md](docs/ALTERNATIVE_PROPOSAL.md) | What we would do differently if we started again, and the measured result of each change. |
| [docs/CHOICE_IMPACT.md](docs/CHOICE_IMPACT.md)   | Legal and ethical limits, business impact of the choices, how a user can misuse them, user experience. |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md)         | Speed, cost of type checking, accuracy on the truth tables and on chess positions, training curves. |
| [docs/INPUT_FORMATS_COMPARAISON.md](docs/INPUT_FORMATS_COMPARAISON.md) | The five input encodings tried for the analyzer, and why King-Rays-Escape (141 inputs) was kept. |
| [CONTRIBUTING.md](CONTRIBUTING.md)               | Code style, commits, CI, how to add an activation function, a task, a label or a hyperparameter. |

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** pip

### Installation for Linux

1. **Clone the repository:**

```sh
git clone git@github.com:pierrick-simon/MyTorch.git
```

2. **Navigate to the project directory:**

```sh
cd MyTorch
```

3. **Compile the project** (creates the four executables at the root of the repository):

```sh
make
```

4. **install dependencies:**

```sh
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

5. **Build a chess training file** (only needed for `my_torch_analyzer`):

```sh
./dataset.sh 2000
```

`dataset.sh` extracts `dataset.zip` into `dataset/` if needed, then writes 9 × 2,000 positions into `train.txt` (see [Chess analyzer](#chess-analyzer)).

The programs use `.venv/bin/python` as interpreter: run them from the root of the repository.

## Usage

### Neural network generator

`my_torch_generator` creates new, untrained neural networks from one or more YAML configuration files and saves them as `.nn` files.

```sh
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

- `config_file_i`: a `.yml` configuration file describing the network to generate.
- `nb_i`: the number of networks to generate from this configuration file.

Each network is saved next to its configuration file, named after it with a `_N.nn` suffix (starting at 1). Each network gets its own random seed (the current timestamp), so networks generated from the same file start with different weights. If the configuration file sets `seed`, network `_N` uses `seed + N - 1` instead: they are still different from each other, but generating them again gives the same networks.

```sh
./my_torch_generator model.yml 3
# -> model_1.nn, model_2.nn, model_3.nn
```

The program exits with code `84` and prints the help if the arguments are invalid (missing count, count that is not a number, file that does not end with `.yml` or does not exist).

### Configuration files

A configuration file is a YAML file that describes the shape of the network. Example (`model.yml`, a network able to learn XOR):

```yaml
nbInputs: 2
nbOutputs: 1
hiderLayersLayout:
  - 2
learningRate: 0.1
activationFunction: Sigmoid
```

| Key                        | Type        | Required  | Description                                                                                                      |
|----------------------------|-------------|-----------|------------------------------------------------------------------------------------------------------------------|
| `nbInputs`                 | `int`       | yes       | Number of inputs of the network.                                                                                 |
| `nbOutputs`                | `int`       | yes       | Number of neurons in the output layer.                                                                           |
| `hiderLayersLayout`        | `list[int]` | yes       | Number of neurons in each hidden layer, from first to last.                                                      |
| `learningRate`             | `float`     | yes       | Learning rate used when training the network.                                                                    |
| `activationFunction`       | `str`       | see below | Activation function of every layer.                                                                              |
| `hiddenActivationFunction` | `str`       | see below | Activation function of the hidden layers (overrides `activationFunction`).                                       |
| `outputActivationFunction` | `str`       | see below | Activation function of the output layer (overrides `activationFunction`).                                        |
| `momentum`                 | `float`     | no        | Share of the previous weight update added to the next one (0 <= momentum < 1). Default: `0` (no momentum).       |
| `initialBias`              | `float`     | no        | Initial bias of every neuron. Default: `0.5`.                                                                    |
| `seed`                     | `int`       | no        | Seed of the first generated network; the next ones use `seed + 1`, `seed + 2`... Default: the current timestamp. |
| `patience`                 | `int`       | no        | Validations in a row below the best before the analyzer offers to stop (>= 1). Default: never offers to stop.    |
| `validationInterval`       | `int`       | no        | Positions trained between two validations, with `patience` (>= 1). Default: `5000`.                              |
| `batchSize`                | `int`       | no        | Samples averaged into each weight update (>= 1). Default: `1` (online training, an update after every sample).   |

- A missing required key stops the generator with an error.
- The activation functions must be given in one of these ways, otherwise the generator stops with an error:
  - `activationFunction` alone: every layer uses it;
  - both `hiddenActivationFunction` and `outputActivationFunction`;
  - `activationFunction` plus one of the two others, which overrides it for its layers.
- `initialBias` and `seed` are optional. Set `seed` to get the same networks every time the generator is run, for example to compare two configurations fairly.
- `momentum` is optional and must be between `0` (included) and `1` (excluded), otherwise the generator stops with an error. Each weight update becomes `Δw ← learningRate × error × input + momentum × Δw (previous)`: a share of the previous update is kept, which smooths the noise of online training and speeds it up along directions where the gradient stays the same. The bias has no momentum. With a high momentum (e.g. `0.9`), the steps are about `1 / (1 - momentum)` times larger, so the learning rate may need to be lowered.
- `validationInterval` without `patience` is ignored and reported with a warning.
- `batchSize` is optional and must be at least `1`, otherwise the generator stops with an error. With `batchSize: B`, the network computes the update of `B` samples with the same weights, then applies their mean once: `Δw ← learningRate × mean(error × input) + momentum × Δw (previous)`. The mean is less noisy than a single sample, but there are `B` times fewer updates per epoch, so the learning rate usually needs to be raised.
- Any other key is ignored and reported with a warning.

For example, a network with 64 inputs, two hidden layers of 32 and 16 neurons and 3 outputs, using ReLU in the hidden layers and Sigmoid in the output layer:

```yaml
nbInputs: 64
nbOutputs: 3
hiderLayersLayout:
  - 32
  - 16
learningRate: 0.05
hiddenActivationFunction: ReLU
outputActivationFunction: Sigmoid
```

The same XOR network as above, using the optional keys to start with zero biases and reproducible weights:

```yaml
nbInputs: 2
nbOutputs: 1
hiderLayersLayout:
  - 2
learningRate: 0.1
activationFunction: Sigmoid
initialBias: 0
seed: 42
```

A complete configuration for the chess analyzer (141 inputs, 5 outputs) using every key, with one `activationFunction` for every layer:

```yaml
nbInputs: 141
nbOutputs: 5
hiderLayersLayout:
  - 32
  - 16
learningRate: 0.01
activationFunction: Sigmoid
momentum: 0.5
initialBias: 0
seed: 1234
patience: 5
validationInterval: 2000
batchSize: 4
```

The same complete configuration, with a different activation function for the hidden layers and for the output layer:

```yaml
nbInputs: 141
nbOutputs: 5
hiderLayersLayout:
  - 32
  - 16
learningRate: 0.01
hiddenActivationFunction: ReLU
outputActivationFunction: Sigmoid
momentum: 0.5
initialBias: 0
seed: 1234
patience: 5
validationInterval: 2000
batchSize: 4
```

### Activation functions

| Name        | Function                     | Output range | Typical use                                                   |
|-------------|------------------------------|--------------|---------------------------------------------------------------|
| `Sigmoid`   | `1 / (1 + e^-x)`             | `]0, 1[`     | Hidden layers of small networks, outputs that are 0 or 1.     |
| `Tanh`      | `tanh(x)`                    | `]-1, 1[`    | Hidden layers, outputs between -1 and 1.                      |
| `ReLU`      | `max(0, x)`                  | `[0, +inf[`  | Hidden layers of deeper networks.                             |
| `LeakyReLU` | `max(0.01x, x)`              | `]-inf, +inf[` | Like ReLU, but neurons keep learning on negative inputs.   |
| `Linear`    | `x`                          | `]-inf, +inf[` | Output layer of a network that predicts a number (regression). |

The output activation must be able to produce the expected values: a `Sigmoid` output can never reach `2`, so a network that must output a count or any number above 1 needs a `Linear` output.

### Chess analyzer

`my_torch_analyzer` trains a network on chess positions, or uses a trained network to predict whether a king is in check or checkmate, and by which side.

```sh
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

- `--train`: train the network `LOADFILE` on the positions of `CHESSFILE`, then save it into `SAVEFILE` if given, otherwise back into `LOADFILE`. Every line must have a label.
- `--predict`: print one prediction per position of `CHESSFILE`: `Nothing`, `Check White`, `Check Black`, `Checkmate White` or `Checkmate Black`.
- `LOADFILE`: a `.nn` file with **141 inputs** and **5 outputs**.
- `CHESSFILE`: one position per line, in FEN notation, optionally followed by its label:

```
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
3r1k1r/pp6/2nb2Qp/q2N3N/2B2Pb1/8/PPP3P1/2KR3R b - - 2 27 Nothing
```

The labels are `Nothing`, `Check White`, `Check Black`, `Checkmate White` and `Checkmate Black` (case insensitive). The colour is the side **giving** the check, and each label is one output of the network. The program exits with code `84` if an argument is invalid, a file cannot be read or a line is not a legal position.

How a training goes:

- The positions are shuffled with the seed of the network, then trained for **20 epochs**. In each epoch, the rare labels are drawn again (at most 4 times each position) so that every label is seen about as often as the most common one.
- A checkpoint `model_N.nn` is saved every 1,000 positions in a directory named after `SAVEFILE` without `.nn` (`chess_trained/` for `--save chess_trained.nn`). The directory is emptied at the start of each training, and the analyzer refuses to start if it contains anything other than `.nn` files.
- If the configuration file of the network sets `patience`, 10 % of `CHESSFILE` (at most 500 positions) is kept aside and used to check the network every `validationInterval` positions (5,000 by default). The best network is saved in `best.nn` in the same directory. If the network gets worse `patience` times in a row, the analyzer offers to stop and keep the best network (see [docs/CHESS_ANALYZER.md](docs/CHESS_ANALYZER.md#validation-during-training)).
- `Ctrl+C` stops the training and saves the network trained so far.

Example, from a shuffled extract of the dataset:

```sh
cat > chess.yml << 'EOF'
nbInputs: 141
nbOutputs: 5
hiderLayersLayout:
  - 64
  - 32
learningRate: 0.01
hiddenActivationFunction: LeakyReLU
outputActivationFunction: Sigmoid
momentum: 0.5
initialBias: 0
patience: 5
validationInterval: 5000
EOF
./dataset.sh 2000 train.txt
./dataset.sh 300 test.txt
./my_torch_generator chess.yml 1
.venv/bin/python -O my_torch_analyzer --train --save chess_trained.nn chess_1.nn train.txt
./my_torch_analyzer --predict chess_trained.nn test.txt
```

This trains `chess_1.nn` and writes `chess_trained.nn`, `chess_trained/model_0.nn`, `chess_trained/model_1000.nn`... and `chess_trained/best.nn`. To see how the network improved from one checkpoint to the next, see [Finding the best model](#from-the-search-to-the-training-chart) (steps 5 and 6).

`dataset.sh` builds a shuffled training file from the dataset:

```sh
./dataset.sh NB [OUTPUT_FILE]
```

- `NB`: number of positions picked at random from **each** of the 9 files of `dataset/` (3 states × 3 piece counts), so the output holds `9 × NB` positions. A file with fewer than `NB` lines is taken whole, with a warning.
- `OUTPUT_FILE`: the file to write. Default: `train.txt`.
- If `dataset/` does not exist, the script extracts `dataset.zip` first.
- The positions are picked without a seed, so two calls give different files, which may share a few positions.

The input encoding, the dataset and training advice are described in [docs/CHESS_ANALYZER.md](docs/CHESS_ANALYZER.md).

## Optimized mode (`-O`)

Every function of the project is decorated with `@typechecked` from `typeguard`, which checks the types of the arguments and of the return value at each call. These checks catch mistakes early, but the network calls small functions millions of times, so they make training **9 to 16 times slower**.

Running Python with the `-O` option (the capital letter O, not zero) disables them. Call the virtual environment's Python explicitly, followed by `-O` and the program:

```sh
.venv/bin/python -O my_torch_analyzer --train chess_1.nn train.txt
.venv/bin/python -O my_torch_analyzer --predict chess_1.nn test.txt
.venv/bin/python -O my_torch_bonus --train hamming hamming_1.nn
.venv/bin/python -O my_torch_generator model.yml 3
.venv/bin/python -O my_torch_hyperoptimizer
```

- With `-O`, `@typechecked` returns the function unchanged, so the code runs exactly as fast as if the decorators were not there.
- `-O` also removes `assert` statements. The project has none, so the results are the same in both modes.
- A type error is no longer reported where it happens, but further down in the computation, or not at all. Use the normal mode (`./my_torch_analyzer ...`) when developing or debugging, and `-O` for long trainings.

| Network         | Training step, normal mode | Training step, `-O` | One pass over the chess dataset, `-O` |
|-----------------|---------------------------:|--------------------:|--------------------------------------:|
| 6-8-1           | 0.138 ms                   | 0.011 ms            | —                                     |
| 141-64-5        | 12.6 ms                    | 0.90 ms             | ≈ 11 min                              |
| 141-64-32-5     | 15.1 ms                    | 1.17 ms             | ≈ 14 min                              |
| 141-128-64-5    | 35.4 ms                    | 2.85 ms             | ≈ 34 min                              |

A training of the analyzer runs 20 epochs over its file: with `./dataset.sh 2000` and a 141-64-32-5 network, that is about 600,000 positions, around 12 minutes with `-O` and about 2 h 30 without.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md#cost-of-type-checking) for all the measures.

## Finding the best model

Training a chess network takes long, so trying configurations by hand is slow. `my_torch_hyperoptimizer` tries many configurations automatically and keeps the best one. This section goes from the search to a chart of the training of the best configuration found.

```sh
.venv/bin/python -O my_torch_hyperoptimizer
```

The program takes no argument: everything is set in [src/my_torch_hyperoptimizer/constants.py](src/my_torch_hyperoptimizer/constants.py). Always run it with `-O` (see [Optimized mode](#optimized-mode--o)): it trains hundreds of thousands of positions.

### How the search works

1. **Data**: `dataset.sh PER_FILE` builds `hyperopt/dataset_PER_FILE.txt` (9,000 positions by default). The file is built only once and reused by the next runs, since `dataset.sh` picks at random: every run compares the configurations on the same positions. The first `VALIDATION_RATIO` (20 %) of the file is the validation set, used only to score the configurations, never to train them.
2. **Random search**: `NB_TRIALS` configurations (27 by default) are drawn at random from `SEARCH_SPACE` in [search_space.py](src/my_torch_hyperoptimizer/search_space.py): hidden layers, learning rate (drawn on a log scale), momentum, hidden activation function, initial bias and batch size. Every network starts from the same `SEED`, so for a same layout the differences of score come from the hyperparameters and not from a lucky initialisation.
3. **Successive halving**: every configuration is trained on `MIN_BUDGET` positions (5,000) and scored on the validation set. Only the best third (`ETA = 3`) is kept, trained `ETA` times longer (it continues from where it stopped, it does not restart), and scored again:

| Rung | Configurations | Positions seen by each |
|------|---------------:|-----------------------:|
| 1    | 27             | 5,000                  |
| 2    | 9              | 15,000                 |
| 3    | 3              | 45,000                 |

Once `ETA` configurations or fewer are left, the best one wins. Bad configurations only cost 5,000 positions each, and only the promising ones get long trainings. The networks are trained the same way as by the analyzer (`balancedEpoch` and `trainBatch` of [train.py](src/my_torch_analyzer/train.py)). A network whose weights explode (`OverflowError`) gets a score of 0 %.

The search takes about 15 to 30 minutes with the default values, depending on the layouts drawn. It writes:

- `hyperopt/results.csv`: one row per configuration and per rung it reached, with its validation accuracy (columns `id,positions,accuracy,` then the hyperparameters). The `id` stays the same from one rung to the next, to follow a configuration.
- `hyperopt/best.yml`: the winning configuration, ready for `my_torch_generator`, with `patience: 5` and `validationInterval: 5000` so the analyzer keeps a validation set during the long training.

The scores of the search come from short trainings: they rank the configurations, they do not tell the accuracy of a fully trained network. The steps below measure it.

### From the search to the training chart

1. **Search the best configuration:**

```sh
make
.venv/bin/python -O my_torch_hyperoptimizer
```

2. **Build a training file and a test file.** The test file is used only to measure the network, on positions it was never trained on:

```sh
./dataset.sh 2000 train.txt
./dataset.sh 300 test.txt
```

3. **Generate the network** of the best configuration:

```sh
./my_torch_generator hyperopt/best.yml 1
# -> hyperopt/best_1.nn
```

4. **Train it.** The analyzer saves a checkpoint `model_N.nn` every 1,000 positions in the directory named after `--save` (`hyperopt_best/` here), plus the best network on its validation set in `best.nn`:

```sh
.venv/bin/python -O my_torch_analyzer --train --save hyperopt_best.nn hyperopt/best_1.nn train.txt
```

5. **Evaluate every checkpoint on the test file.** This writes `hyperopt_best/checkpoints.csv` (columns `iteration,accuracy,loss`, and a last row `best` for `best.nn`):

```sh
.venv/bin/python -O benchmarks/checkpoints_csv.py hyperopt_best test.txt
```

6. **Draw the chart** of the improvement during the training: the accuracy and the loss on the test file at each checkpoint, with the `best` row as a dashed line:

```sh
.venv/bin/python benchmarks/checkpoints_plot.py hyperopt_best/checkpoints.csv -o docs/images/hyperoptimizer.png
```

The `best` row of `checkpoints.csv` is the final result: the accuracy on the test file of the network the analyzer kept.

- `checkpoints.csv` is not a `.nn` file, so a new `--train` with the same `--save` refuses to overwrite the directory: delete `checkpoints.csv` first.
- `dataset.sh` picks at random from the same files each time, so `test.txt` may share a few positions with `train.txt` or with the search dataset, which makes the test score a little too high.
- To search other values, edit `SEARCH_SPACE` in [search_space.py](src/my_torch_hyperoptimizer/search_space.py); the columns of `results.csv` follow it. To search longer, raise `NB_TRIALS` (keep a power of `ETA`: 9, 27, 81) or `MIN_BUDGET` in [constants.py](src/my_torch_hyperoptimizer/constants.py). Delete `hyperopt/dataset_*.txt` to draw new positions.

## Bonus

### Truth-table trainer

`my_torch_bonus` trains a network on a small truth table, or measures how well it has learned it.

```sh
./my_torch_bonus [--train | --eval] TASK FILE [--iteration N]
```

- `--train`: train the network `FILE` on `TASK` for `N` iterations (default `100000`), then save it into `FILE`.
- `--eval`: test the network `FILE` on `N` random cases of `TASK` (default `1000`) and print its accuracy.
- `TASK`: the truth table to use (see below).
- `FILE`: a `.nn` file, for example made by `my_torch_generator`.

| Task      | Inputs | Outputs | Expected output                                                         | Tolerance |
|-----------|--------|---------|-------------------------------------------------------------------------|-----------|
| `and`     | 2      | 1       | `1` if both inputs are `1`, else `0`.                                   | `0.1`     |
| `xor`     | 2      | 1       | `1` if the inputs are different, else `0`.                              | `0.1`     |
| `hamming` | 6      | 1       | Number of positions where the first 3 bits differ from the last 3 (`0` to `3`). | `0.5` |

- The network must have the number of inputs and outputs of the task.
- An output counts as correct when it is closer to the expected value than the tolerance. For `hamming`, a tolerance of `0.5` means the output, rounded to the nearest integer, is the right distance.
- Pressing `Ctrl+C` during training pauses it and asks what to do: `save` (or `s`, the default) saves the network trained so far, `resume` (or `r`) continues training, `quit` (or `q`) exits without saving.
- The program exits with code `84` and prints the help if the arguments are invalid.

### End-to-end example: learning the Hamming distance

This example creates a network, trains it to compute the Hamming distance between two 3-bit words, and evaluates it. Run it from the project directory, after the [installation](#installation-for-linux).

1. **Write the configuration file** `hamming.yml`. The hidden layer uses `Sigmoid`, and the output layer uses `Linear` so that the network can output a distance of `2` or `3`:

```sh
cat > hamming.yml << 'EOF'
nbInputs: 6
nbOutputs: 1
hiderLayersLayout:
  - 8
learningRate: 0.05
hiddenActivationFunction: Sigmoid
outputActivationFunction: Linear
EOF
```

2. **Generate the network:**

```sh
./my_torch_generator hamming.yml 1
```

3. **Train it** (about 30 seconds):

```sh
./my_torch_bonus --train hamming hamming_1.nn
```

4. **Evaluate it:**

```sh
./my_torch_bonus --eval hamming hamming_1.nn
```
