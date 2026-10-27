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
python3.12 -m venv .venv
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
```

| Key                 | Type        | Description                                                      |
|---------------------|-------------|------------------------------------------------------------------|
| `nbInputs`          | `int`       | Number of inputs of the network.                                 |
| `nbOutputs`         | `int`       | Number of neurons in the output layer.                           |
| `hiderLayersLayout` | `list[int]` | Number of neurons in each hidden layer, from first to last.      |
| `learningRate`      | `float`     | Learning rate used when training the network.                    |

- All four keys are **required**: a missing key stops the generator with an error.
- Any other key is ignored and reported with a warning.
- All neurons use the sigmoid activation function.

For example, a network with 64 inputs, two hidden layers of 32 and 16 neurons and 3 outputs:

```yaml
nbInputs: 64
nbOutputs: 3
hiderLayersLayout:
  - 32
  - 16
learningRate: 0.05
```
