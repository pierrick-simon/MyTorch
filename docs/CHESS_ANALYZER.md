# Chess analyzer

`my_torch_analyzer` trains a neural network to tell, from a chess position, whether a king is in check. This document describes the input files, how a position is turned into numbers for the network, and how to train a network on the provided dataset.

- [Usage](#usage)
- [Input file format](#input-file-format)
- [Input encoding](#input-encoding)
- [Output encoding](#output-encoding)
- [Dataset](#dataset)
- [Training on the dataset](#training-on-the-dataset)

## Usage

```sh
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

| Argument          | Description                                                                                  |
|-------------------|----------------------------------------------------------------------------------------------|
| `--train`         | Train the network `LOADFILE` on every position of `CHESSFILE`, then save it.                  |
| `--predict`       | Print one prediction per position of `CHESSFILE`.                                             |
| `--save SAVEFILE` | With `--train` only: save the trained network into `SAVEFILE` instead of `LOADFILE`.          |
| `LOADFILE`        | A `.nn` file, for example made by `my_torch_generator`.                                       |
| `CHESSFILE`       | A file with one position per line (see below).                                                |
| `--help`, `-h`    | Print the help and exit with `0`.                                                             |

The program exits with code `84` when the arguments are invalid, when a file cannot be read, when a line of `CHESSFILE` is not a legal position, or when a line has no label in training mode.

Pressing `Ctrl+C` during training stops it and saves the network trained so far. Pressing it during prediction stops printing.

## Input file format

Each line holds a position in [Forsyth–Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) (6 fields separated by spaces), optionally followed by the expected label:

```
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
3r1k1r/pp6/2nb2Qp/q2N3N/2B2Pb1/8/PPP3P1/2KR3R b - - 2 27 Nothing
```

Accepted labels (case insensitive): `Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black`. The colour is the side **giving** the check.

Every line is checked with [`python-chess`](https://python-chess.readthedocs.io/): a line with fewer than 6 FEN fields, an illegal position or an unknown label stops the program with code `84` and the reason.

## Input encoding

Each position becomes a vector of **783** numbers between `0` and `1`, so a network for the analyzer must have `nbInputs: 783`.

| Part             | Size | Encoding                                                                                       |
|------------------|------|------------------------------------------------------------------------------------------------|
| Pieces           | 768  | 64 squares × 12 values: one-hot of the piece on the square (6 piece types × 2 colours), all `0` on an empty square |
| Side to move     | 1    | `1` if white plays, `0` if black plays                                                          |
| Castling rights  | 4    | one value per right, in the order `K`, `Q`, `k`, `q`                                            |
| En passant       | 8    | one-hot of the file (`a` to `h`) of the en passant square, all `0` if there is none             |
| Halfmove clock   | 1    | `min(halfmove, 100) / 100`                                                                      |
| Fullmove number  | 1    | `min(fullmove, 200) / 200`                                                                      |

Squares are read in FEN order: from `a8` to `h8`, then rank 7, down to `h1`. For each square, the 12 values are ordered king, queen, rook, bishop, knight, pawn, first for black then for white.

> At commit `adc7c32`, `Color.getColor` returns `BLACK` for every piece, so white pieces are encoded in the black values and the two colours cannot be told apart. [BENCHMARKS.md](BENCHMARKS.md#learning) measures the effect of fixing it.

Why this encoding:

- **One-hot pieces** rather than one number per square (for example `1` for a pawn, `9` for a queen): a number would tell the network that a queen is "nine pawns", an order that does not exist for the question "is the king attacked?". With one-hot values each piece type has its own weights for every square, which is what the network needs to learn which pieces attack which squares.
- **Every FEN field is kept**, even those that do not change whether a king is in check (castling, en passant, clocks): they cost 15 inputs out of 783, and keep the encoding useful for other questions about the same positions.
- **Values between 0 and 1**: the clocks are divided by a maximum so that no input is much larger than the others, which would make its weights dominate the first updates.

The encoding is done in [`Chessboard.py`](../src/my_torch_analyzer/Chessboard.py).

## Output encoding

The network has **2 outputs**:

| Output | Label                                                |
|--------|------------------------------------------------------|
| `0`    | `Nothing`                                            |
| `1`    | `Check` (any of the four check and checkmate labels) |

In training, the expected output is `[1, 0]` or `[0, 1]`. In prediction, the program prints the label of the output with the highest value, `Nothing` or `Check`. A network for the analyzer must therefore have `nbOutputs: 2`; with any other number, training stops with an error because the outputs and the expected outputs do not have the same length.

A `Sigmoid` output layer fits this encoding: each output is between 0 and 1, like the expected values.

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
unzip dataset.zip    # creates dataset/
```

## Training on the dataset

The analyzer trains in **one pass, in file order**. Since each file of the dataset holds a single label, training on one file after the other would teach the network to always answer the last label it saw. Mix and shuffle the files first, and keep some positions aside to check the network on positions it has never seen:

```sh
cat dataset/*/*.txt | shuf > all.txt
head -n 20000 all.txt > test.txt
tail -n +20001 all.txt > train.txt
```

Write a configuration with 783 inputs and 2 outputs, for example `chess.yml`:

```yaml
nbInputs: 783
nbOutputs: 2
hiderLayersLayout:
  - 32
learningRate: 0.05
hiddenActivationFunction: Sigmoid
outputActivationFunction: Sigmoid
```

Then generate, train and predict:

```sh
./my_torch_generator chess.yml 1
./my_torch_analyzer --train --save chess_trained.nn chess_1.nn train.txt
./my_torch_analyzer --predict chess_trained.nn test.txt > predictions.txt
```

To train for several passes (epochs), run `--train` again on the trained network, ideally after shuffling `train.txt` again.

The whole file is parsed before training starts, and each position takes about 17 KB of memory once encoded: the full dataset would need about 12 GB. Train on chunks instead, one after the other, with the same network:

```sh
split -l 50000 train.txt chunk_
for f in chunk_*; do
    .venv/bin/python -O my_torch_analyzer --train chess_1.nn "$f"
done
```

Training speed depends a lot on the size of the network and on how Python is run; see [BENCHMARKS.md](BENCHMARKS.md) before choosing a network for the full dataset.
