# Benchmarks

This document measures how fast MyTorch is, how well it learns the truth tables of `my_torch_bonus`, and what it achieves on chess positions. Every number can be reproduced with the scripts of [`benchmarks/`](../benchmarks/).

- [Setup](#setup)
- [Speed](#speed)
- [Cost of type checking](#cost-of-type-checking)
- [Truth tables](#truth-tables)
- [Chess positions](#chess-positions)
- [Summary](#summary)

## Setup

| Item       | Value                                         |
|------------|-----------------------------------------------|
| CPU        | Intel Core Ultra 7 255H (one core used)       |
| Python     | 3.14.3                                        |
| typeguard  | 4.6.0                                         |
| Code       | commit `adc7c32`                              |
| Date       | 2026-10-28                                    |

Unless stated otherwise, benchmarks run with `python -O`, which disables the `typeguard` checks (see [below](#cost-of-type-checking)).

```sh
.venv/bin/python -O benchmarks/speed.py      # a few seconds
.venv/bin/python -O benchmarks/tables.py     # about 1 minute
.venv/bin/python -O benchmarks/chess_dataset.py      # about 4 minutes, needs dataset/
```

## Speed

Time of one forward pass (`tick`) and of one training step (`tick` then `learn`), for the networks of the truth tables and for chess-sized networks:

| Network      | Shape            | Parameters | Forward (ms) | Training step (ms) | Steps / s |
|--------------|------------------|-----------:|-------------:|-------------------:|----------:|
| XOR          | 2-2-1            | 9          | 0.001        | 0.003              | 290,535   |
| Hamming      | 6-8-1            | 65         | 0.003        | 0.011              | 88,975    |
| Chess small  | 783-32-2         | 25,154     | 0.814        | 1.963              | 509       |
| Chess medium | 783-64-32-2      | 52,322     | 1.709        | 4.192              | 239       |
| Chess large  | 783-256-64-32-3  | 219,331    | 7.120        | 17.728             | 56        |

The time grows linearly with the number of parameters (about 80 ns per parameter per training step for the chess networks), as expected from loops that visit every weight once forward and once backward.

What this means for one pass over the 723,019 positions of the chess dataset:

| Network      | With `-O`  | Without `-O` |
|--------------|-----------:|-------------:|
| 783-32-2     | ≈ 24 min   | ≈ 3 h 05     |
| 783-64-32-2  | ≈ 50 min   | ≈ 6 h 30     |
| 783-256-64-32-3 | ≈ 3 h 35 | ≈ 28 h       |

## Cost of type checking

Same measure, with and without the runtime checks of `typeguard`:

| Network          | Training step, checks on (ms) | Training step, `-O` (ms) | Slowdown |
|------------------|------------------------------:|-------------------------:|---------:|
| 2-2-1            | 0.048                         | 0.003                    | × 16     |
| 6-8-1            | 0.138                         | 0.011                    | × 12.5   |
| 783-32-2         | 15.40                         | 1.77                     | × 8.7    |
| 783-64-32-2      | 32.45                         | 3.75                     | × 8.7    |
| 783-256-64-32-3  | 138.3                         | 15.5                     | × 8.9    |

`typeguard` checks every argument of every call, including every element of the input lists, and the network makes one call per neuron per sample. The checks catch type errors early during development (see [DESIGN_CHOICES.md](DESIGN_CHOICES.md#runtime-type-checking-with-typeguard)), but for long trainings `-O` should always be used:

```sh
.venv/bin/python -O my_torch_analyzer --train chess_1.nn train.txt
.venv/bin/python -O my_torch_bonus --train hamming hamming_1.nn
```

## Truth tables

Each configuration is trained 5 times, with seeds 1 to 5, on random rows of the table. A seed is **solved** when every row of the table is within the tolerance of the task (`0.1` for AND and XOR, `0.5` for Hamming, see the [README](../README.md#truth-table-trainer)). The mean accuracy is the share of rows within the tolerance, averaged over the 5 seeds.

### After 100,000 iterations (the default of `my_torch_bonus --train`)

| Task    | Shape  | Learning rate | Hidden / output     | Mean accuracy | Seeds solved | Time (s) |
|---------|--------|--------------:|---------------------|--------------:|-------------:|---------:|
| AND     | 2-2-1  | 0.1           | Sigmoid / Sigmoid   | 100.0 %       | 5/5          | 0.3      |
| XOR     | 2-2-1  | 0.1           | Sigmoid / Sigmoid   | 100.0 %       | 5/5          | 0.3      |
| XOR     | 2-4-1  | 0.1           | Sigmoid / Sigmoid   | 100.0 %       | 5/5          | 0.5      |
| XOR     | 2-4-1  | 0.1           | Tanh / Sigmoid      | 100.0 %       | 5/5          | 0.5      |
| XOR     | 2-4-1  | 0.1           | ReLU / Sigmoid      | 100.0 %       | 5/5          | 0.5      |
| XOR     | 2-4-1  | 0.1           | LeakyReLU / Sigmoid | 100.0 %       | 5/5          | 0.5      |
| Hamming | 6-8-1  | 0.05          | Sigmoid / Sigmoid   | 37.5 %        | 0/5          | 1.1      |
| Hamming | 6-8-1  | 0.05          | Sigmoid / Linear    | 100.0 %       | 5/5          | 1.1      |
| Hamming | 6-8-1  | 0.05          | Tanh / Linear       | 100.0 %       | 5/5          | 1.1      |
| Hamming | 6-8-1  | 0.01          | ReLU / Linear       | 100.0 %       | 5/5          | 1.1      |
| Hamming | 6-16-1 | 0.05          | Sigmoid / Linear    | 100.0 %       | 5/5          | 2.0      |

### Convergence: seeds solved after N iterations

| Task    | Shape  | Hidden / output     | 1,000 | 10,000 | 30,000 | 100,000 |
|---------|--------|---------------------|------:|-------:|-------:|--------:|
| AND     | 2-2-1  | Sigmoid / Sigmoid   | 0/5   | 0/5    | 5/5    | 5/5     |
| XOR     | 2-2-1  | Sigmoid / Sigmoid   | 0/5   | 0/5    | 2/5    | 5/5     |
| XOR     | 2-4-1  | Sigmoid / Sigmoid   | 0/5   | 0/5    | 3/5    | 5/5     |
| XOR     | 2-4-1  | Tanh / Sigmoid      | 0/5   | 4/5    | 5/5    | 5/5     |
| XOR     | 2-4-1  | ReLU / Sigmoid      | 0/5   | 4/5    | 5/5    | 5/5     |
| XOR     | 2-4-1  | LeakyReLU / Sigmoid | 0/5   | 4/5    | 5/5    | 5/5     |
| Hamming | 6-8-1  | Sigmoid / Sigmoid   | 0/5   | 0/5    | 0/5    | 0/5     |
| Hamming | 6-8-1  | Sigmoid / Linear    | 0/5   | 2/5    | 5/5    | 5/5     |
| Hamming | 6-8-1  | Tanh / Linear       | 0/5   | 5/5    | 5/5    | 5/5     |
| Hamming | 6-8-1  | ReLU / Linear       | 0/5   | 4/5    | 4/5    | 5/5     |
| Hamming | 6-16-1 | Sigmoid / Linear    | 0/5   | 5/5    | 5/5    | 5/5     |

Reproduce a column with `.venv/bin/python -O benchmarks/tables.py N`.

### What we learn

- **The backpropagation is correct.** XOR is not linearly separable, so it can only be solved if the error reaches the hidden layer correctly. Every tested configuration solves it.
- **The output activation decides what can be learnt.** With a `Sigmoid` output, the Hamming network is stuck at 37.5 %: its output stays below 1, so distances 2 and 3 are always wrong, and the network settles on an output close to 1, right only on the 24 rows out of 64 at distance 1. A `Linear` output solves it with the same hidden layer. This is why configuration files can choose the output activation separately.
- **The hidden activation decides how fast it is learnt.** `Tanh`, `ReLU` and `LeakyReLU` hidden layers solve XOR about 3 times sooner than `Sigmoid` (most seeds at 10,000 iterations instead of 30,000 to 100,000). `Sigmoid`'s derivative is at most `0.25`, which shrinks the error sent back to the hidden layer; `Tanh`'s goes up to `1`.
- **More hidden neurons also help.** With 16 hidden neurons instead of 8, the Hamming network is solved by every seed after 10,000 iterations.
- **The default of 100,000 iterations is enough** for every configuration that can solve its task, in about one second with `-O`.

## Chess positions

### Parsing

`fen_parser` checks every line with `python-chess` before encoding it:

| Measure                          | Value                     |
|----------------------------------|---------------------------|
| Parsing speed                    | ≈ 19,600 positions / s    |
| Full dataset (723,019 positions) | ≈ 37 s                    |
| Memory per encoded position      | ≈ 17 KB                   |
| Full dataset in memory           | ≈ 12 GB                   |

Parsing is negligible next to training (37 s against 24 min for one pass of the smallest network), but memory is not: large training files should be split into chunks (see [CHESS_ANALYZER.md](CHESS_ANALYZER.md#training-on-the-dataset)).

### Learning

Networks are trained on a balanced sample of the dataset: 1,000 random positions from each of the 9 files, so one third `Nothing` and two thirds `Check` or `Checkmate`. 7,200 positions are used for training, 1,800 for testing. Since 65.8 % of the test positions are `Check`, a network that always answers `Check` already scores **65.8 %**: this is the baseline to beat.

**Activation functions and depth**, 3 epochs, learning rate 0.01:

| Shape         | Hidden / output   | Test accuracy | Training time (s) |
|---------------|-------------------|--------------:|------------------:|
| 783-32-2      | Sigmoid / Sigmoid | 64.6 %        | 44                |
| 783-32-2      | Tanh / Sigmoid    | 64.9 %        | 42                |
| 783-32-2      | ReLU / Sigmoid    | 64.6 %        | 43                |
| 783-64-32-2   | Sigmoid / Sigmoid | 65.9 %        | 90                |

**Learning rate and piece colours**, 783-32-2, Sigmoid / Sigmoid, test accuracy after N epochs (share of test positions predicted `Check` in brackets):

| Piece colours in the input | Learning rate | 1 epoch        | 5 epochs       | 10 epochs      |
|----------------------------|--------------:|---------------:|---------------:|---------------:|
| Not distinguished (`adc7c32`) | 0.01       | 63.4 % (94 %)  | 65.4 % (96 %)  | 66.2 % (96 %)  |
| Distinguished              | 0.002         | 63.2 % (93 %)  | 64.3 % (95 %)  | 64.6 % (95 %)  |
| Distinguished              | 0.01          | 63.5 % (90 %)  | 65.7 % (94 %)  | 66.3 % (95 %)  |
| Distinguished              | 0.05          | 65.6 % (94 %)  | 67.3 % (86 %)  | **68.8 %** (79 %) |

At commit `adc7c32`, `Color.getColor` returns `BLACK` for every piece, so white and black pieces use the same 6 of the 12 values of each square. The "distinguished" rows replace it with the intended function (`WHITE` for upper-case letters) to measure the effect of the fix.

Reproduce the last row with `.venv/bin/python -O benchmarks/chess_dataset.py 10 0.05` (the network sees the samples in another order than in the table above, so the numbers differ slightly).

What we learn:

- **On 7,200 positions, the networks barely beat the baseline.** Most answer `Check` for almost every position. Training accuracy stays close to test accuracy, so the problem is not over-fitting: the networks have not found how to detect a check yet.
- **Detecting a check is hard for a small network.** A check depends on the relation between two squares that can be far apart (a rook or a bishop on the other side of the board, with nothing in between). With one-hot squares, each such pattern has to be learnt separately for every pair of squares, which needs far more than 7,200 examples.
- **Distinguishing the colours and a larger learning rate help.** With both, accuracy keeps going up after 10 epochs (68.8 %) and the network starts answering `Nothing`, which none of the other configurations does.
- **The next steps** are to train on much more data (hundreds of thousands of positions, which the [speed measures](#speed) put at tens of minutes per pass with `-O`), to fix the colour encoding, and to keep a learning rate around 0.05.

## Summary

| Question                                         | Answer                                                              |
|--------------------------------------------------|---------------------------------------------------------------------|
| Does the backpropagation work?                   | Yes: AND, XOR and Hamming are learnt perfectly by every seed.       |
| Which activation functions to use?               | `Tanh` or `ReLU` in hidden layers; an output that can reach the expected values (`Linear` for counts). |
| How long does a training take?                   | About 80 ns per parameter per sample with `-O`: 24 min per pass of the dataset for a 783-32-2 network. |
| Should training use `-O`?                        | Yes: type checking makes it 9 to 16 times slower.                   |
| Does the chess analyzer work?                    | It trains and predicts, but on 7,200 positions it barely beats always answering `Check` (68.8 % against 65.8 %). It needs much more data. |
