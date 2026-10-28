# MyTorch

### Development

By *Ariel Amriou*(ariel.amriou@epitech.eu) & *Pierrick Simon*(pierrick.simon@epitech.eu).

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

3. **install dependencies:**

```sh
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Neural network generator

`my_torch_generator` creates new, untrained neural networks from one or more YAML configuration files and saves them as `.nn` files.

```sh
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

- `config_file_i`: a `.yml` configuration file describing the network to generate.
- `nb_i`: the number of networks to generate from this configuration file.

Each network is saved next to its configuration file, named after it with a `_N.nn` suffix (starting at 1). Each network gets its own random seed (the current timestamp), so networks generated from the same file start with different weights.

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

| Key                        | Type        | Required | Description                                                                  |
|----------------------------|-------------|----------|------------------------------------------------------------------------------|
| `nbInputs`                 | `int`       | yes      | Number of inputs of the network.                                             |
| `nbOutputs`                | `int`       | yes      | Number of neurons in the output layer.                                       |
| `hiderLayersLayout`        | `list[int]` | yes      | Number of neurons in each hidden layer, from first to last.                  |
| `learningRate`             | `float`     | yes      | Learning rate used when training the network.                                |
| `activationFunction`       | `str`       | see below | Activation function of every layer.                                         |
| `hiddenActivationFunction` | `str`       | see below | Activation function of the hidden layers (overrides `activationFunction`).  |
| `outputActivationFunction` | `str`       | see below | Activation function of the output layer (overrides `activationFunction`).   |

- A missing required key stops the generator with an error.
- The activation functions must be given in one of these ways, otherwise the generator stops with an error:
  - `activationFunction` alone: every layer uses it;
  - both `hiddenActivationFunction` and `outputActivationFunction`;
  - `activationFunction` plus one of the two others, which overrides it for its layers.
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

### Activation functions

| Name        | Function                     | Output range | Typical use                                                   |
|-------------|------------------------------|--------------|---------------------------------------------------------------|
| `Sigmoid`   | `1 / (1 + e^-x)`             | `]0, 1[`     | Hidden layers of small networks, outputs that are 0 or 1.     |
| `Tanh`      | `tanh(x)`                    | `]-1, 1[`    | Hidden layers, outputs between -1 and 1.                      |
| `ReLU`      | `max(0, x)`                  | `[0, +inf[`  | Hidden layers of deeper networks.                             |
| `LeakyReLU` | `max(0.01x, x)`              | `]-inf, +inf[` | Like ReLU, but neurons keep learning on negative inputs.   |
| `Linear`    | `x`                          | `]-inf, +inf[` | Output layer of a network that predicts a number (regression). |

The output activation must be able to produce the expected values: a `Sigmoid` output can never reach `2`, so a network that must output a count or any number above 1 needs a `Linear` output.

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
