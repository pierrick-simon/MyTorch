# Design choices

This document explains the main technical decisions of MyTorch, the alternatives we considered, and what each choice costs. Numbers quoted here come from [BENCHMARKS.md](BENCHMARKS.md).

- [Pure Python, no NumPy](#pure-python-no-numpy)
- [One object per neuron](#one-object-per-neuron)
- [Online stochastic gradient descent](#online-stochastic-gradient-descent)
- [Separate hidden and output activation functions](#separate-hidden-and-output-activation-functions)
- [Derivatives computed from the output](#derivatives-computed-from-the-output)
- [Weight initialisation](#weight-initialisation)
- [Seeds](#seeds)
- [YAML configuration files](#yaml-configuration-files)
- [Pickle for the `.nn` format](#pickle-for-the-nn-format)
- [Runtime type checking with typeguard](#runtime-type-checking-with-typeguard)
- [Validating chess positions with python-chess](#validating-chess-positions-with-python-chess)
- [Two output classes for the analyzer](#two-output-classes-for-the-analyzer)
- [Pluggable bonus tasks](#pluggable-bonus-tasks)
- [Error handling and exit codes](#error-handling-and-exit-codes)

## Pure Python, no NumPy

**Choice.** The network is written with Python lists and loops. The only external libraries are for input/output (`pyyaml`, `python-chess`, `click`) and type checking (`typeguard`); none of them does any computation for the network.

**Why.** The goal of the project is to build a neural network from scratch. With plain loops, every multiplication of the forward pass and every update of the backpropagation is visible in [`perceptron.py`](../src/my_torch/perceptron.py) and [`MLP.py`](../src/my_torch/MLP.py), and can be read next to the formulas in [ARCHITECTURE.md](ARCHITECTURE.md#backpropagation).

**Cost.** Speed. A forward and backward pass over a 783-32-2 network takes about 1.8 ms with optimisations (`python -O`), where a matrix library would take a few microseconds. This limits the size of the networks and the number of chess positions that can be used in a reasonable time. Moving the inner loops of `Perceptron.tick` and `Perceptron.learn` to vectors would be the first step if speed became the priority.

## One object per neuron

**Choice.** A network is a list of layers, and a layer is a list of `Perceptron` objects, each one holding its weights, bias, learning rate, last input and last output.

**Why.** It maps the vocabulary of the course (a perceptron, a layer of perceptrons) directly onto the code, and it makes each neuron testable on its own. Storing `lastInput` and `lastOutput` in the neuron means `learn` needs no extra argument: the forward pass leaves behind exactly what the backward pass needs.

**Cost.** Each neuron is a Python object with its own list, so there is one attribute lookup and one function call per neuron and per sample. It also makes the network stateful: `learn` must be called right after the `tick` of the same sample.

## Online stochastic gradient descent

**Choice.** The weights are updated after every sample. There are no mini-batches, no momentum and no learning rate schedule.

**Why.** It is the simplest form of gradient descent, it uses no extra memory, and it lets training be stopped at any sample (with `Ctrl+C`) with a usable network. On the truth tables, it is enough: XOR, AND and the 3-bit Hamming distance are learnt perfectly by every tested seed (see [BENCHMARKS.md](BENCHMARKS.md#truth-tables)).

**Cost.** The gradient of a single sample is noisy, so the learning rate must stay small and the order of the samples matters. The analyzer trains in file order, so its training file must be shuffled (see [CHESS_ANALYZER.md](CHESS_ANALYZER.md#training-on-the-dataset)).

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

**Choice.** Weights are drawn uniformly in `[-1, 1]`, biases start at `0.5` unless the configuration file sets `initialBias`.

**Why.** It is the simplest initialisation that breaks the symmetry between neurons, and it works well for the small networks of the truth tables.

**Cost.** The range does not depend on the number of inputs of the neuron. With 783 inputs and deep `ReLU` layers, the weighted sums grow from layer to layer: with the 783-256-64-32 shape of our first chess configuration, a `ReLU` network made `math.exp` overflow in the `Sigmoid` output on the first sample. Scaling the range by `1/√nbInputs` (Xavier or He initialisation) would fix this; until then, prefer `Sigmoid` or `Tanh` hidden layers, or a single hidden layer, for large inputs.

## Seeds

**Choice.** The generator gives each network the current timestamp as seed, or `seed + N - 1` for copy `N` when the configuration file sets `seed`. The seed is saved inside the `.nn` file.

**Why.** Several networks generated from the same configuration start with different weights, so one can generate a few and keep the best. Since the seed is saved, the same network can be rebuilt, and `my_torch_bonus --eval` uses it to draw the same test cases every time for a given network, so two evaluations of the same file give the same score. Setting `seed` in the configuration file makes a whole run reproducible, which is what comparisons between configurations need.

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
- **Size**: each weight is a full Python float. A 783-256-64-32-3 network weighs about 2 MB.

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

## Two output classes for the analyzer

**Choice.** The analyzer has two outputs, `Nothing` and `Check`; the four check and checkmate labels are grouped into `Check`.

**Why.** Detecting that a king is attacked is the first problem to solve, and with two balanced classes the accuracy of the network is easy to read. The labels are mapped in `STATE_LABELS` in [`Chessboard.py`](../src/my_torch_analyzer/Chessboard.py): separating checkmate from check, or the colour of the attacker, only means adding entries to the `State` enum and to this table, and using a network with more outputs.

## Pluggable bonus tasks

**Choice.** Each truth table of `my_torch_bonus` is a module of `src/bonus/tables/` with a `TABLE`, a `train` and an `evaluate` function. The task given on the command line is imported by name.

**Why.** Adding a task (OR, a parity function, a larger Hamming distance) is one new file, with no change to the argument parsing or the training loop. Each task chooses its own tolerance: `0.1` for the binary tables, `0.5` for the Hamming distance, which is the same as rounding the output to the nearest integer.

## Error handling and exit codes

**Choice.** Invalid arguments and invalid files make the programs print a message and exit with code `84`; `--help` exits with `0`.

**Why.** It is the convention of the project's specification, and it lets scripts tell a usage error from a success. The analyzer catches `ValueError` and `OSError` around the parsing and the training, so a malformed line or a missing file gives a one-line message rather than a Python traceback.

The generator does not catch the errors of `parse_yml` yet: an invalid configuration file stops it with a traceback and exit code `1`. The message at the bottom of the traceback still explains what is wrong.
