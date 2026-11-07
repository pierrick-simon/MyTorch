# Chess analyzer

`my_torch_analyzer` trains a neural network to tell, from a chess position, whether a king is in check or checkmate, and which side gives it. This document describes the input files, how a position is turned into numbers for the network, how a training goes, and how to train a network on the provided dataset.

- [Usage](#usage)
- [Input file format](#input-file-format)
- [Input encoding](#input-encoding)
- [Output encoding](#output-encoding)
- [Dataset](#dataset)
- [How a training goes](#how-a-training-goes)
- [Training on the dataset](#training-on-the-dataset)

## Usage

```sh
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

| Argument          | Description                                                                                  |
|-------------------|----------------------------------------------------------------------------------------------|
| `--train`         | Train the network `LOADFILE` on the positions of `CHESSFILE` (20 epochs), then save it.       |
| `--predict`       | Print one prediction per position of `CHESSFILE`.                                             |
| `--save SAVEFILE` | With `--train` only: save the trained network into `SAVEFILE` instead of `LOADFILE`.          |
| `LOADFILE`        | A `.nn` file, for example made by `my_torch_generator`.                                       |
| `CHESSFILE`       | A file with one position per line (see below).                                                |
| `--help`, `-h`    | Print the help and exit with `0`.                                                             |

The program exits with code `84` when the arguments are invalid, when a file cannot be read, when a line of `CHESSFILE` is not a legal position, or when a line has no label in training mode.

Pressing `Ctrl+C` during training stops it and saves the network trained so far (or the best one, if validation found a better one). Pressing it during prediction stops printing.

In training mode, the analyzer also writes a directory named after `SAVEFILE` without `.nn` (or after `LOADFILE` when `--save` is not given), with a checkpoint every 1,000 positions and, with validation, the best network: see [Checkpoints](#checkpoints).

### Validation during training

Validation only runs for networks generated with a `patience` key in their configuration file. Without it, every position of `CHESSFILE` is trained on and nothing below happens.

Before training, the analyzer keeps 10 % of `CHESSFILE` aside, at most 500 positions, drawn at random with the seed of the network. These positions are never trained on: they measure how the network does on positions it has not seen.

Every `validationInterval` training positions (a key of the configuration file, 5,000 by default), the analyzer computes the accuracy on these positions (the share of positions where the highest output is the expected label) and shows it next to the progress bar, with the best accuracy so far. Each time the best is beaten, the network is saved in `<SAVEFILE without .nn>/best.nn`.

If the accuracy stays below the best for `patience` validations in a row, the network is getting worse on new positions, usually because it is over-fitting or its learning rate is too high. The analyzer then asks whether to stop:

- `y` (the default): training stops, and the best network is saved into `SAVEFILE` instead of the last one;
- `n`: training goes on, and the analyzer asks again after `patience` more validations below the best.

When the standard input is not a terminal, the analyzer cannot ask: it prints a warning and goes on. At the end of training, it prints the final and the best validation accuracies, and where the best network is.

The analyzer offers to stop at the earliest after `patience × validationInterval` positions without improvement: 25,000 with `patience: 5` and the default interval. Both values are stored in the `.nn` file when the network is generated. Networks generated before they existed do not have them: generate them again to train them. The size of the validation set is a constant at the top of [`src/my_torch_analyzer/train.py`](../src/my_torch_analyzer/train.py): `VALIDATION_SIZE`. A file of fewer than 10 positions has no validation.

## Input file format

Each line holds a position in [Forsyth–Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) (6 fields separated by spaces), optionally followed by the expected label:

```
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
3r1k1r/pp6/2nb2Qp/q2N3N/2B2Pb1/8/PPP3P1/2KR3R b - - 2 27 Nothing
```

Accepted labels (case insensitive): `Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black`. The colour is the side **giving** the check.

Every line is checked with [`python-chess`](https://python-chess.readthedocs.io/): a line with fewer than 6 FEN fields, an illegal position or an unknown label stops the program with code `84` and the reason.

## Input encoding

Each position becomes a vector of **141** numbers between `0` and `1`, so a network for the analyzer must have `nbInputs: 141`. The encoding is called **King-Rays-Escape**: instead of describing the 64 squares, it describes the board as seen from the king of the side to move, since a check is a property of that king.

| Group                | Inputs | Content                                                                                   |
|----------------------|-------:|-------------------------------------------------------------------------------------------|
| Rays                 | 104    | 8 directions from the king × (first piece met: 12 kinds, own/enemy × 6 types, + its distance / 7) |
| Knight squares       | 8      | An enemy knight is on the square, one input per knight jump from the king                 |
| Pawn squares         | 2      | An enemy pawn is on a square that attacks the king                                        |
| Neighbouring squares | 16     | 8 squares around the king × (off the board, own piece on it)                              |
| Escape squares       | 8      | 8 squares around the king × (attacked by the enemy, with the king removed from the board) |
| King                 | 3      | King row / 7, king column / 7, `1` if white plays                                         |

- Pieces are *own* or *enemy* (relative to the side to move), never white or black, so the same pattern means the same thing for both colours.
- The escape squares are tested with the king removed: a rook giving check along a line also attacks the square behind the king, which would otherwise look like a safe retreat.
- The castling rights, the en passant square and the move counters of the FEN line are not used: they do not change whether the king is attacked.
- The input never contains the answer: the network still has to learn how these facts combine, for example whether a checking piece can be captured or blocked.

This encoding was chosen after comparing five of them with the same network and the same data; [INPUT_FORMATS_COMPARAISON.md](INPUT_FORMATS_COMPARAISON.md) describes them, the attack test, and the training curves of each. The previous encoding (783 inputs: 64 squares × 12 piece planes plus the other FEN fields) is the one measured in [BENCHMARKS.md](BENCHMARKS.md#learning).

The encoding is done in [`chessboard.py`](../src/my_torch_analyzer/chessboard.py): `NB_INPUTS` gives its size and `_kingRelativeInput` builds the vector.

## Output encoding

The network has **5 outputs**, one per label:

| Output | Label             |
|--------|-------------------|
| `0`    | `Nothing`         |
| `1`    | `Check White`     |
| `2`    | `Check Black`     |
| `3`    | `Checkmate White` |
| `4`    | `Checkmate Black` |

In training, the expected output is `1` for the label of the position and `0` for the four others. In prediction, the program prints the label of the output with the highest value. A network for the analyzer must therefore have `nbOutputs: 5`; with any other number, training stops with an error because the outputs and the expected outputs do not have the same length.

A `Sigmoid` output layer fits this encoding: each output is between 0 and 1, like the expected values. The labels are listed in `STATE_LABELS` and the `State` enum of [`chessboard.py`](../src/my_torch_analyzer/chessboard.py).

## Dataset

`dataset.zip` contains 723,019 labelled positions, split by label and by number of pieces on the board:

| File                          | Positions | Label                         |
|-------------------------------|----------:|-------------------------------|
| `nothing/10_pieces.txt`       | 100,000   | `Nothing`                     |
| `nothing/20_pieces.txt`       | 100,000   | `Nothing`                     |
| `nothing/many_pieces.txt`     | 100,000   | `Nothing`                     |
| `check/10_pieces.txt`         | 75,198    | `Check White` / `Check Black` |
| `check/20_pieces.txt`         | 100,000   | `Check White` / `Check Black` |
| `check/many_pieces.txt`       | 74,102    | `Check White` / `Check Black` |
| `checkmate/10_pieces.txt`     | 45,087    | `Checkmate White` / `Checkmate Black` |
| `checkmate/20_pieces.txt`     | 100,000   | `Checkmate White` / `Checkmate Black` |
| `checkmate/many_pieces.txt`   | 28,632    | `Checkmate White` / `Checkmate Black` |

```sh
unzip dataset.zip    # creates dataset/, done automatically by dataset.sh
```

In a file built by `dataset.sh`, a third of the positions are `Nothing`, a third `Check White` or `Check Black` and a third `Checkmate White` or `Checkmate Black`: the analyzer oversamples the rarer labels during training (see [How a training goes](#how-a-training-goes)).

## How a training goes

1. `CHESSFILE` is parsed and encoded in full before the training starts (about 0.13 ms and 1.6 KB per position with `-O`: the whole dataset would take about 1.2 GB).
2. The positions are shuffled with the seed of the network. With `patience`, the validation positions are set aside (see [Validation during training](#validation-during-training)).
3. The network is trained for **20 epochs** (`EPOCHS` in [`train.py`](../src/my_torch_analyzer/train.py)). Each epoch is a new shuffle of the training positions where the rarer labels are drawn again, with replacement, until they are as many as the most common one, without taking any position more than 4 times (`MAX_OVERSAMPLING`). The network does not learn to always answer `Nothing` just because it is the most common label.
4. The positions are sent one by one, or by groups of `batchSize` when the configuration sets it.
5. At the end, the last network is saved into `SAVEFILE`, or the best one if validation found a better one.

### Checkpoints

The analyzer saves the network every 1,000 positions (`CHECKPOINT_EVERY`), starting before the first one: `model_0.nn`, `model_1000.nn`, `model_2000.nn`... in the directory named after `SAVEFILE` without `.nn`. With `patience`, the best network on the validation set is also saved there as `best.nn`.

- The directory is emptied at the start of each training. If it contains a file that is not a `.nn` file, the analyzer refuses to start, so that it never deletes something it did not write. If a file (not a directory) already has its name, it refuses as well.
- Each checkpoint is a normal `.nn` file: it can be used with `--predict`, or trained again.
- [`benchmarks/checkpoints_csv.py`](../benchmarks/checkpoints_csv.py) evaluates every checkpoint on the same positions and writes `checkpoints.csv` in the directory; [`benchmarks/checkpoints_plot.py`](../benchmarks/checkpoints_plot.py) draws its accuracy and loss curves. See [BENCHMARKS.md](BENCHMARKS.md#training-curves). Remove `checkpoints.csv` before training again with the same `SAVEFILE`.

## Training on the dataset

`dataset.sh` builds a training file with the same number of positions from each of the 9 files of the dataset:

```sh
./dataset.sh 2000 train.txt    # 18,000 positions
./dataset.sh 300 test.txt      # 2,700 positions to test the network
```

It picks the positions at random without a seed, so `test.txt` may share a few positions with `train.txt`. To get a test file with no position in common, split one shuffled file instead:

```sh
cat dataset/*/*.txt | shuf > all.txt
head -n 20000 all.txt > test.txt
tail -n +20001 all.txt | head -n 50000 > train.txt
```

Write a configuration with 141 inputs and 5 outputs, for example `chess.yml` (the configuration used to compare the input encodings):

```yaml
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
seed: 42
patience: 5
validationInterval: 5000
```

With `patience` and `validationInterval`, the analyzer checks the network every 5,000 positions on positions it does not train on, and offers to stop if it gets worse 5 times in a row (see [Validation during training](#validation-during-training)).

Then generate, train and predict:

```sh
./my_torch_generator chess.yml 1
.venv/bin/python -O my_torch_analyzer --train --save chess_trained.nn chess_1.nn train.txt
.venv/bin/python -O my_torch_analyzer --predict chess_trained.nn test.txt > predictions.txt
```

To measure the accuracy on `test.txt`, compare `predictions.txt` with the labels of `test.txt`, or run `benchmarks/checkpoints_csv.py chess_trained test.txt`, whose `best` row is the accuracy of `best.nn`.

To train longer, run `--train` again on the trained network, with another `--save` so that the checkpoints of the first training are not deleted. To choose the configuration itself (layers, learning rate, activation functions...), `my_torch_hyperoptimizer` tries many of them automatically: see [Finding the best model](../README.md#finding-the-best-model).

Training speed depends a lot on the size of the network and on how Python is run (always use `-O` for long trainings); see [BENCHMARKS.md](BENCHMARKS.md#speed) before choosing a network for the full dataset.
