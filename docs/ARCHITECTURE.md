# Architecture

This document explains how MyTorch is organised and how a neural network works inside it, from the YAML configuration file to the trained `.nn` file.

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [The `my_torch` library](#the-my_torch-library)
- [Forward pass](#forward-pass)
- [Backpropagation](#backpropagation)
- [The `.nn` file format](#the-nn-file-format)
- [The programs](#the-programs)

## Overview

MyTorch is made of one library and three programs built on top of it:

```
                 config.yml
                     │
                     ▼
           ┌───────────────────┐
           │ my_torch_generator │  builds untrained networks
           └───────────────────┘
                     │  model_1.nn, model_2.nn, ...
                     ▼
     ┌───────────────────────────────┐
     │                               │
     ▼                               ▼
┌───────────────────┐       ┌────────────────┐
│ my_torch_analyzer │       │ my_torch_bonus │
│  chess positions  │       │  truth tables  │
│  --train/--predict│       │  --train/--eval│
└───────────────────┘       └────────────────┘
     │                               │
     └──────────────┬────────────────┘
                    ▼
          src/my_torch (MLP, Perceptron, activation functions)
```

Every program loads and saves networks the same way (`MLP.load` / `MLP.save`), so a network made by the generator can be trained by either program.

## Repository layout

```
.
├── my_torch_generator          # entry point → src/my_torch_generator/main.py
├── my_torch_analyzer           # entry point → src/my_torch_analyzer/main.py
├── my_torch_bonus              # entry point → src/bonus/main.py
├── requirements.txt
├── dataset.zip                 # chess positions (see CHESS_ANALYZER.md)
├── docs/                       # this documentation
├── benchmarks/                 # scripts used for BENCHMARKS.md
└── src/
    ├── my_torch/               # the neural network library
    │   ├── MLP.py              # multi-layer perceptron: forward, backprop, save/load
    │   ├── perceptron.py       # one neuron: weights, bias, update rule
    │   └── activationClasses/  # Sigmoid, Tanh, ReLU, LeakyReLU, Linear
    ├── my_torch_generator/
    │   ├── main.py             # argument checks, one network per requested copy
    │   └── parse_yml.py        # configuration validation
    ├── my_torch_analyzer/
    │   ├── main.py             # argument parsing (argparse), error handling
    │   ├── fen_parser.py       # reads and validates a file of FEN lines
    │   ├── Chessboard.py       # FEN → 783 input values, label → expected output
    │   ├── train.py            # one pass of online training over the file
    │   └── predict.py          # prints one prediction per line
    └── bonus/
        ├── main.py             # argument checks, loads the task module
        ├── train.py            # generic training loop with Ctrl+C menu
        ├── evaluate.py         # generic accuracy measure
        └── tables/             # one module per task: and, xor, hamming
```

The three executables at the root are three-line Python scripts that call the `main()` of their package and exit with its return value. They use `.venv/bin/python` as interpreter, so they must be run from the root of the repository.

## The `my_torch` library

### `Perceptron`

A perceptron is one neuron. It stores:

| Attribute        | Meaning                                                            |
|------------------|--------------------------------------------------------------------|
| `weights`        | one weight per input, initialised uniformly in `[-1, 1]`           |
| `bias`           | initialised to `0.5`                                               |
| `lr`             | learning rate, copied from the configuration                       |
| `activationFunc` | the `activation` function of its activation class                  |
| `lastInput`      | the inputs of the last `tick`, needed to update the weights        |
| `lastOutput`     | the output of the last `tick`, needed to compute the error         |

`tick(inputs)` computes `activation(Σ inputs[i] × weights[i], bias)` and remembers its input and output. `learn(unitError)` applies the update rule (see [Backpropagation](#backpropagation)).

### `MLP`

`MLP(nbInputs, layers, nbOutputs, learningRate, (hiddenActivation, outputActivation), seed)` builds a list of layers, each layer being a list of `Perceptron`:

- one layer per entry of `hiderLayersLayout`, using the hidden activation function;
- one output layer of `nbOutputs` perceptrons, using the output activation function.

The `seed` is given to Python's `random` module before the weights are drawn, so the same configuration and the same seed always produce the same network. The generator uses the current timestamp as seed, so every generated network is different.

### Activation classes

An activation class is a class with two static functions:

| Function                 | Used by          | Meaning                                                       |
|--------------------------|------------------|---------------------------------------------------------------|
| `activation(sum, bias)`  | forward pass     | output of the neuron for a weighted sum of inputs and a bias   |
| `derivation(output)`     | backpropagation  | derivative of the activation, **expressed from its output**   |

| Class       | `activation(s, b)`     | `derivation(y)`             |
|-------------|------------------------|-----------------------------|
| `Sigmoid`   | `1 / (1 + e^-(s+b))`   | `y × (1 − y)`               |
| `Tanh`      | `tanh(s + b)`          | `1 − y²`                    |
| `ReLU`      | `max(0, s + b)`        | `1 if y > 0 else 0`         |
| `LeakyReLU` | `max(0.01(s+b), s+b)`  | `1 if y > 0 else 0.01`      |
| `Linear`    | `s + b`                | `1`                         |

The derivative is written from the output `y` rather than from the input `s + b`, because the output is what each perceptron already remembers (`lastOutput`). This works for all five functions above; see [DESIGN_CHOICES.md](DESIGN_CHOICES.md#derivatives-computed-from-the-output) for why.

The names accepted in configuration files are the keys of `ActivationClassNames` in [`src/my_torch/activationClasses/__init__.py`](../src/my_torch/activationClasses/__init__.py). Adding an activation function means adding one file with these two functions and one entry in this dictionary.

## Forward pass

`MLP.tick(inputs)` sends the inputs to every perceptron of the first layer, then the list of their outputs to every perceptron of the next layer, and so on. The outputs of the last layer are returned.

```
inputs ──► layer 0 ──► layer 1 ──► ... ──► output layer ──► outputs
          (hidden act.)                    (output act.)
```

The number of inputs must match the number of weights of the first layer, otherwise `Perceptron.tick` raises a `ValueError`.

## Backpropagation

`MLP.learn(outputs, targets)` performs one step of stochastic gradient descent on the squared error `E = ½ Σ (target − output)²`, for the sample that was just sent to `tick`.

1. **Output layer.** For each output neuron `k`:

   ```
   δk = f'out(ok) × (tk − ok)
   ```

2. **Hidden layers**, from the last one to the first. For a neuron `j` whose outputs go to the neurons `k` of the next layer:

   ```
   δj = f'hidden(oj) × Σk δk × wkj
   ```

   All the `δ` are computed before any weight changes, so the error is propagated with the weights that produced the output.

3. **Update.** Every neuron updates its weights and bias with its own `δ` and the inputs it received:

   ```
   wi   ← wi + lr × δ × xi
   bias ← bias + lr × δ
   ```

Since `δ` is `−∂E/∂(s+b)`, adding `lr × δ × xi` moves each weight against the gradient of the error.

Training is **online**: the weights change after every sample. There is no batch, no momentum and no learning rate decay.

## The `.nn` file format

A `.nn` file is the `MLP` object serialised with Python's `pickle` module. It contains the full state of the network: every weight and bias, the learning rate, the seed and the two activation classes.

- `MLP.save(path)` asks for confirmation before overwriting an existing file, unless `forceOverwrite=True` (used after training).
- `MLP.load(path)` returns the saved `MLP`.

Because pickle stores classes by their module path, the files only load if the modules keep their names (`src.my_torch.MLP`, `src.my_torch.activationClasses.sigmoid`, ...). This is why `MLP.py` keeps its name despite the naming rules of the linter. See [DESIGN_CHOICES.md](DESIGN_CHOICES.md#pickle-for-the-nn-format) for the trade-offs of this format.

> Loading a pickle file can execute arbitrary code. Only load `.nn` files that you or a trusted person generated.

## The programs

### `my_torch_generator`

1. `checkParam` checks that the arguments come in pairs `config.yml count`, that each file ends with `.yml` and exists, and that each count is a number.
2. `parse_yml` loads the YAML file and `check_config` validates it: required keys, types, unknown keys (removed with a warning) and activation functions (resolved to their class).
3. For each requested copy, an `MLP` is built with the timestamp as seed and saved as `<config>_<N>.nn`.

### `my_torch_analyzer`

1. `argparse` reads `--predict` or `--train [--save SAVEFILE]`, `LOADFILE` and `CHESSFILE`. Any error exits with code `84`.
2. `fen_parser` reads `CHESSFILE` line by line. Each line is checked by `python-chess` (legal position, known label) and converted to a `Chessboard`.
3. `--train`: every line must have a label. The network sees each position once, in file order, then is saved to `SAVEFILE` (or back to `LOADFILE`). `Ctrl+C` stops the training early and still saves.
4. `--predict`: one line is printed per position, the label of the output with the highest value.

The input encoding is described in [CHESS_ANALYZER.md](CHESS_ANALYZER.md).

### `my_torch_bonus`

1. `checkParam` checks the mode, that `TASK` is a module of `src/bonus/tables/`, the `.nn` file and the optional `--iteration N`.
2. The task module is imported by name with `importlib`. Each task module defines a `TABLE` of `{inputs, expectedOutputs}` and two functions, `train` and `evaluate`, that call the generic ones with the right tolerance.
3. `--train` draws random rows of the table `N` times and calls `tick` then `learn`. `Ctrl+C` opens a menu to save, resume or quit.
4. `--eval` draws `N` random rows and prints the share of outputs closer to the expected value than the tolerance.

Adding a task means adding one file in `src/bonus/tables/`, with no change to `main.py`.
