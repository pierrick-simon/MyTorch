# Design choices

This document explains the main technical decisions of MyTorch, the alternatives we considered, and what each choice costs. Numbers quoted here come from [BENCHMARKS.md](BENCHMARKS.md).

- [Pure Python, no NumPy](#pure-python-no-numpy)
- [One object per neuron](#one-object-per-neuron)
- [Online stochastic gradient descent, optional mini-batches](#online-stochastic-gradient-descent-optional-mini-batches)
- [Separate hidden and output activation functions](#separate-hidden-and-output-activation-functions)
- [Derivatives computed from the output](#derivatives-computed-from-the-output)
- [Weight initialisation](#weight-initialisation)
- [Seeds](#seeds)
- [YAML configuration files](#yaml-configuration-files)
- [Pickle for the `.nn` format](#pickle-for-the-nn-format)
- [Runtime type checking with typeguard](#runtime-type-checking-with-typeguard)
- [Validating chess positions with python-chess](#validating-chess-positions-with-python-chess)
- [A king-relative input encoding](#a-king-relative-input-encoding)
- [Five output classes for the analyzer](#five-output-classes-for-the-analyzer)
- [Balanced epochs](#balanced-epochs)
- [Checkpoints during training](#checkpoints-during-training)
- [Random search and successive halving](#random-search-and-successive-halving)
- [Pluggable bonus tasks](#pluggable-bonus-tasks)
- [Error handling and exit codes](#error-handling-and-exit-codes)

## Pure Python, no NumPy

**Choice.** The network is written with Python lists and loops. The only external libraries are for input/output (`pyyaml`, `python-chess`, `click`, `tqdm` for the progress bar), type checking (`typeguard`) and the charts of the benchmarks (`matplotlib`); none of them does any computation for the network.

**Why.** The goal of the project is to build a neural network from scratch. With plain loops, every multiplication of the forward pass and every update of the backpropagation is visible in [`perceptron.py`](../src/my_torch/perceptron.py) and [`MLP.py`](../src/my_torch/MLP.py), and can be read next to the formulas in [ARCHITECTURE.md](ARCHITECTURE.md#backpropagation).

**Cost.** Speed. A forward and backward pass over the 141-64-32-5 network of the analyzer takes about 1.2 ms with optimisations (`python -O`), and 1.8 ms for the former 783-32-2 network, where a matrix library would take a few microseconds. This limits the size of the networks and the number of chess positions that can be used in a reasonable time. Moving the inner loops of `Perceptron.tick` and `Perceptron.learn` to vectors would be the first step if speed became the priority.

## One object per neuron

**Choice.** A network is a list of layers, and a layer is a list of `Perceptron` objects, each one holding its weights, bias, learning rate, last input and last output.

**Why.** It maps the vocabulary of the course (a perceptron, a layer of perceptrons) directly onto the code, and it makes each neuron testable on its own. Storing `lastInput` and `lastOutput` in the neuron means `learn` needs no extra argument: the forward pass leaves behind exactly what the backward pass needs.

**Cost.** Each neuron is a Python object with its own list, so there is one attribute lookup and one function call per neuron and per sample. It also makes the network stateful: `learn` must be called right after the `tick` of the same sample.

## Online stochastic gradient descent, optional mini-batches

**Choice.** By default the weights are updated after every sample. A configuration can set `batchSize` to update them once every `batchSize` samples with the mean of their updates. There is no learning rate schedule. An optional `momentum` (`0` by default) keeps a share of the previous update of each weight.

**Why.** It is the simplest form of gradient descent, it uses no extra memory, and it lets training be stopped at any sample (with `Ctrl+C`) with a usable network. On the truth tables, it is enough: XOR, AND and the 3-bit Hamming distance are learnt perfectly by every tested seed (see [BENCHMARKS.md](BENCHMARKS.md#truth-tables)).

**Cost.** The gradient of a single sample is noisy, so the learning rate must stay small and the order of the samples matters. The analyzer shuffles its positions again at every epoch, with the seed of the network, so the order of the training file does not matter (see [Balanced epochs](#balanced-epochs)).

**Mini-batches.** Averaging the updates of a batch gives a less noisy direction, and the samples of a batch all use the same weights, so their updates could be computed in parallel. The cost is one extra sum per weight in every neuron, and `batchSize` times fewer updates per epoch: the learning rate usually has to be raised. `batchSize: 1` keeps the online code path, so it gives exactly the same networks as before. With `Ctrl+C`, the samples of the unfinished batch are still applied. Since the number of trained positions grows by a whole batch at a time, checkpoints and validations happen on the first batch that reaches their interval.

## Separate hidden and output activation functions

**Choice.** A configuration can give one `activationFunction` for every layer, or a `hiddenActivationFunction` and an `outputActivationFunction`.

**Why.** The two kinds of layers have different jobs:

- the **output** activation sets the range of values the network can produce. A `Sigmoid` output can never reach `2`, so the Hamming distance cannot be learnt with it: it stays at 37.5 % accuracy, while the same network with a `Linear` output reaches 100 %.
- the **hidden** activation sets how fast the network learns. On XOR, after 30,000 iterations, `Tanh`, `ReLU` and `LeakyReLU` hidden layers solve all 5 seeds, while `Sigmoid` solves 3 out of 5.

A single function for the whole network would force a bad compromise in one of the two places. `activationFunction` alone is still accepted, so the simple configurations stay simple.

## Derivatives computed from the output

**Choice.** Each activation class gives its derivative as a function of the neuron **output** `y`: `y(1 − y)` for Sigmoid, `1 − y²` for Tanh, `1 if y > 0` for ReLU.

**Why.** The output is already stored by every neuron during the forward pass, so the backward pass does not need to keep or recompute the weighted sums. For Sigmoid and Tanh, the derivative written from the output is also cheaper: no exponential to compute again.

**Cost.** This only works for functions whose derivative can be written from the output, which is the case for all five current functions (for ReLU, LeakyReLU and Linear, `y > 0` exactly when `s + b > 0`). A function such as Swish or GELU could not be added without also storing the weighted sum.

## Weight initialisation

**Choice.** Weights are drawn uniformly in `[-limit, limit]`, where the limit depends on the activation function of the layer and on its size: `√(6 / nbInputs)` for `ReLU` and `LeakyReLU` (He initialisation), `√(6 / (nbInputs + nbOutputs))` for the others (Xavier initialisation). Biases start at `0.5` unless the configuration file sets `initialBias`.

**Why.** With a fixed range such as `[-1, 1]`, the weighted sum of a neuron grows with its number of inputs. With 783 inputs, `Sigmoid` and `Tanh` neurons start saturated (their derivative is close to `0`, so they barely learn), and deep `ReLU` networks made `math.exp` overflow in the `Sigmoid` output on the first sample. Scaling the range keeps the weighted sums, and the errors sent back, of the same order from layer to layer. `ReLU` cancels half of its inputs, so He initialisation uses a range about `√2` times larger than Xavier.

**Cost.** Networks generated with the same `seed` before this change have other weights: generate them again to compare configurations.

## Seeds

**Choice.** The generator gives each network the current timestamp as seed, or `seed + N - 1` for copy `N` when the configuration file sets `seed`. The seed is saved inside the `.nn` file.

**Why.** Several networks generated from the same configuration start with different weights, so one can generate a few and keep the best. Since the seed is saved, the same network can be rebuilt, and the programs reuse it wherever they need randomness: `my_torch_bonus --eval` draws the same test cases every time for a given network, so two evaluations of the same file give the same score, and the analyzer uses it to shuffle the positions and pick its validation set, so training the same network on the same file twice gives the same result. Setting `seed` in the configuration file makes a whole run reproducible, which is what comparisons between configurations need.

## YAML configuration files

**Choice.** Networks are described in YAML, read with `yaml.safe_load`, then checked key by key.

**Why.** YAML is easy to read and write by hand, supports lists for the layer layout, and `safe_load` cannot run code from the file. Checking the keys gives clear messages for common mistakes: a missing required key or an unknown activation function stops the generator, and an unknown key (often a typo) is reported with a warning.

The key that lists the hidden layers is named `hiderLayersLayout`. It is kept with this spelling so that existing configuration files stay valid.

## Pickle for the `.nn` format

**Choice.** A `.nn` file is the `MLP` object saved with `pickle`.

**Why.** It saves the whole network, including the activation classes, in one call, without writing or maintaining a parser. Any attribute added to `MLP` or `Perceptron` is saved automatically.

**Cost.**

- **Security**: loading a pickle file can run arbitrary code. Only trusted `.nn` files should be loaded.
- **Coupling to the code**: classes are stored by module path, so renaming a module (for example `MLP.py`, which the linter would like in lowercase) makes every existing `.nn` file unloadable. This is why that file keeps its name.
- **Size**: each weight is a full Python float, saved with its last update (`lastDelta`) and its batch sum (`deltaSum`). A 141-64-32-5 network weighs about 300 KB, a 783-256-64-32-3 network about 2 MB.

A text format (JSON with the layout, the activation names and the weights) would remove the first two problems; it would be the right choice if networks had to be shared between people or versions.

## Runtime type checking with typeguard

**Choice.** Every function is decorated with `@typechecked`, so the type annotations are checked at each call.

**Why.** Wrong types are the most common mistakes in a project without a compiler: a string from the command line instead of an int, a YAML value of the wrong type, a list of lists instead of a list. With typeguard, they fail at the function that received them, with the name of the argument, instead of further down in the computation.

**Cost.** typeguard checks every element of every list at every call, and the network calls small functions millions of times. Training is **9 to 16 times slower** with the checks (see [BENCHMARKS.md](BENCHMARKS.md#cost-of-type-checking)). typeguard disables its checks when Python runs with `-O`, so long trainings can be run with:

```sh
.venv/bin/python -O my_torch_analyzer --train chess_1.nn train.txt
```

## Validating chess positions with python-chess

**Choice.** Every line of a chess file is loaded with `chess.Board` and rejected if the position is illegal or the label unknown.

**Why.** A wrong line in a training file silently teaches the network something false. Checking legality (two kings, no pawn on the last rank, side not to move not in check, ...) by hand would be long and error prone, and `python-chess` is a well-tested reference. `parse_line` can also compare the label with the real state of the position (`check_consistency=True`).

**Cost.** Parsing is slower than a plain split of the FEN string (see [BENCHMARKS.md](BENCHMARKS.md#chess-positions)). It stays small next to the training time.

The network itself never sees `python-chess`: its inputs are built from the FEN string only (see [CHESS_ANALYZER.md](CHESS_ANALYZER.md#input-encoding)).

## A king-relative input encoding

**Choice.** The analyzer turns a position into 141 numbers that describe the board as seen from the king of the side to move: the first piece met on each of the 8 rays from the king, the knights and pawns that could attack it, and the state of the 8 squares around it, including whether the enemy attacks them (King-Rays-Escape, see [CHESS_ANALYZER.md](CHESS_ANALYZER.md#input-encoding)).

**Why.** The first encoding, 64 squares × 12 piece planes plus the other FEN fields (783 inputs), left the network to find out which distant squares attack the king. With it, the network barely beat always answering the most common label (see [BENCHMARKS.md](BENCHMARKS.md#learning)). Five encodings were then compared with the same network and the same data, and the king-relative one learnt best and fastest: about 96 % against 60 to 66 % for the raw-board encodings ([BENCHMARKS.md](BENCHMARKS.md#input-formats), [INPUT_FORMATS_COMPARAISON.md](INPUT_FORMATS_COMPARAISON.md)). It is also 5.5 times smaller, so each training step is faster and each encoded position takes about 1.6 KB of memory instead of 17 KB.

**Cost.** The encoding knows some chess: it computes which squares are attacked, which a network on the raw board would have to learn. It does not describe pins, captures of the checking piece or blocks, so the network still has to guess them from the rays. It also only fits questions about the king of the side to move.

## Five output classes for the analyzer

**Choice.** The analyzer has one output per label of the dataset: `Nothing`, `Check White`, `Check Black`, `Checkmate White` and `Checkmate Black`.

**Why.** The labels of the dataset give the state of the position and the side giving the check. With one output per label, the prediction is the label of the highest output, with no rule to add afterwards. The labels are mapped in `STATE_LABELS` and the `State` enum of [`chessboard.py`](../src/my_torch_analyzer/chessboard.py). The first versions had two outputs (`Nothing` and `Check`), then three (`Nothing`, `Check`, `Checkmate`), to start from an easier problem.

**Cost.** The classes are not balanced: in a file built by `dataset.sh`, `Nothing` is twice as frequent as each of the four others, and on the full dataset `Checkmate Black` is the rarest. See [Balanced epochs](#balanced-epochs).

## Balanced epochs

**Choice.** The analyzer trains for 20 epochs. Each epoch is a new shuffle of the training positions, where the positions of the rarer labels are drawn again (with replacement) until each label is as frequent as the most common one, with at most 4 copies of a position on average.

**Why.** With online training, a network sees the most common label most often and learns to answer it by default. Oversampling the others gives every label the same weight in the updates, without throwing positions away. Shuffling inside the analyzer, with the seed of the network, makes the result independent of the order of the file and reproducible.

**Cost.** A rare position can be seen several times per epoch, which makes over-fitting on it easier; the limit of 4 copies (`MAX_OVERSAMPLING`) keeps a label with very few positions from being repeated too much. The number of epochs is a constant (`EPOCHS` in [`train.py`](../src/my_torch_analyzer/train.py)), not a key of the configuration file: the validation and `patience` are the way to stop earlier.

## Checkpoints during training

**Choice.** The analyzer saves the network every 1,000 positions (`model_N.nn`) and, with validation, the best one (`best.nn`), in a directory named after the save file. It empties this directory at the start of each training, and refuses if it holds something other than `.nn` files.

**Why.** A training of several minutes or hours should not have to be run again to know how the network evolved: the checkpoints can be evaluated afterwards on any test file (`benchmarks/checkpoints_csv.py`) and drawn as curves (`benchmarks/checkpoints_plot.py`), or one of them can be kept if the end of the training made things worse. Refusing to empty a directory with other files protects files that the analyzer did not write.

**Cost.** Disk space: one `.nn` file per 1,000 positions (about 300 KB each for a 141-64-32-5 network, so about 180 MB for a training of 600,000 positions). The `checkpoints.csv` written by the benchmark script in the same directory must be removed before training again with the same save file.

## Random search and successive halving

**Choice.** `my_torch_hyperoptimizer` draws 27 configurations at random, trains them all on 5,000 positions, keeps the best third, trains them 3 times longer, and so on until 3 or fewer are left. Everything is set in constants, with no command line.

**Why.**

- **Random search rather than a grid**: with 7 hyperparameters, a grid of only 3 values each is already 2,187 trainings. A random draw tries a new value of every hyperparameter at each configuration, so the ones that matter (usually the learning rate) are explored much better for the same number of trainings. The learning rate is drawn on a log scale, since 0.001 and 0.01 differ as much as 0.01 and 0.1.
- **Successive halving**: most bad configurations are already bad after a short training. Stopping them early leaves the budget to the promising ones: 27 × 5,000 + 9 × 10,000 + 3 × 30,000 additional positions, instead of 27 × 45,000 for a full training of each.
- **Same seed for every configuration, same data for every run**: the differences of score come from the hyperparameters, not from a lucky initialisation or a lucky sample of positions.
- **No command line**: the search has many settings that are changed together and rarely; one commented file ([constants.py](../src/my_torch_hyperoptimizer/constants.py)) is simpler to read and edit than a dozen options.

**Cost.** A configuration that learns slowly but ends up better can be eliminated in the first round. The scores come from short trainings, so they rank the configurations but do not give the accuracy of a fully trained network. The search runs in one process, one configuration after the other.

## Pluggable bonus tasks

**Choice.** Each truth table of `my_torch_bonus` is a module of `src/bonus/tables/` with a `TABLE`, a `train` and an `evaluate` function. The task given on the command line is imported by name.

**Why.** Adding a task (OR, a parity function, a larger Hamming distance) is one new file, with no change to the argument parsing or the training loop. Each task chooses its own tolerance: `0.1` for the binary tables, `0.5` for the Hamming distance, which is the same as rounding the output to the nearest integer.

## Error handling and exit codes

**Choice.** Invalid arguments and invalid files make the programs print a message and exit with code `84`; `--help` exits with `0`.

**Why.** It is the convention of the project's specification, and it lets scripts tell a usage error from a success. The analyzer catches `ValueError` and `OSError` around the parsing and the training, so a malformed line or a missing file gives a one-line message rather than a Python traceback.

The generator does not catch the errors of `parse_yml` yet: an invalid configuration file stops it with a traceback and exit code `1`. The message at the bottom of the traceback still explains what is wrong.
