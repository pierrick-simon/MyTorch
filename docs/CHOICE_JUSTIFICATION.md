# Choice justification

This document justifies the major decisions of MyTorch: for each one, what we chose, why, which alternatives we considered, and the evidence that made us decide. The last section links these choices to the constraints of the project. The full technical detail of each choice is in [DESIGN_CHOICES.md](DESIGN_CHOICES.md); the numbers come from [BENCHMARKS.md](BENCHMARKS.md) and from `hyperopt/results.csv`.

- [Language: Python](#language-python)
- [Hidden activation: LeakyReLU rather than Sigmoid](#hidden-activation-leakyrelu-rather-than-sigmoid)
- [Output activation and output classes](#output-activation-and-output-classes)
- [Input encoding](#input-encoding)
- [Size of the network and the other hyperparameters](#size-of-the-network-and-the-other-hyperparameters)
- [Learning algorithm](#learning-algorithm)
- [Avoiding over-fitting](#avoiding-over-fitting)
- [Network and configuration files](#network-and-configuration-files)
- [Reproducibility](#reproducibility)
- [Input validation and errors](#input-validation-and-errors)
- [Link with the constraints of the project](#link-with-the-constraints-of-the-project)

## Language: Python

**Choice.** The whole project is written in Python 3, and the network itself in plain Python (lists and loops, no computing library).

**Why.**

- **Readability of the algorithm.** With plain loops, every multiplication of the forward pass and every update of the backpropagation is visible in [`perceptron.py`](../src/my_torch/perceptron.py) and [`MLP.py`](../src/my_torch/MLP.py), next to the formulas of [ARCHITECTURE.md](ARCHITECTURE.md#backpropagation). We can explain and debug each step.
- **Development speed.** No compilation, no memory management, and libraries for everything around the network: YAML (`pyyaml`), chess rules (`python-chess`), command line (`click`), progress bars (`tqdm`), charts (`matplotlib`). Our time went into the network and the experiments, not into parsers.
- **One language for everything.** The module, the four programs, the benchmarks and the charts share the same code: the hyperoptimizer reuses the training functions of the analyzer, and the benchmark scripts load the same `.nn` files.
- **Team.** Both of us already knew Python well, which mattered with a fixed deadline.

**Alternatives.**

| Alternative | What it would give | Why we did not choose it |
|-------------|--------------------|--------------------------|
| C or C++ | Training many times faster (Python spends about 100 ns per weight per training step). | Much more code for the same features (memory, parsing, charts). The time saved in training would have been spent in development. |
| Python with NumPy | Vectorised layers, much faster training. | The computation is hidden in matrix products, so the backpropagation is no longer readable step by step. It is the first change we would make if speed became the priority. |

**Cost.** Training is slow: about 14 minutes per pass over the dataset for a 141-64-32-5 network with `-O`. We compensated with a smaller input encoding and with a hyperoptimizer that stops bad configurations early.

## Hidden activation: LeakyReLU rather than Sigmoid

**Choice.** The hidden layers of the analyzer use `LeakyReLU`. The module offers `Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU` and `Linear`, and the hidden and output functions are chosen separately.

**Evidence.** We compared the hidden activations three ways:

1. **On XOR** (2-4-1 network, 5 seeds, [BENCHMARKS.md](BENCHMARKS.md#convergence-seeds-solved-after-n-iterations)): after 10,000 iterations, `Tanh`, `ReLU` and `LeakyReLU` solve 4 seeds out of 5, `Sigmoid` solves none. `Sigmoid` needs 30,000 to 100,000 iterations.
2. **On chess, with the hyperoptimizer** (27 random configurations, `hyperopt/results.csv`): the 7 configurations with a `Sigmoid` hidden layer reach at best 55.4 % after 5,000 positions, against 77.6 % for the best one (`LeakyReLU`). None of them is among the 9 kept for the second round, which are all `ReLU`, `LeakyReLU` or `Tanh`.
3. **In theory**: the derivative of `Sigmoid` is at most 0.25, so the error sent back is divided by at least 4 at each layer and the first layers barely learn (vanishing gradient). The derivative of `ReLU` is 1 for every active neuron.

**Why LeakyReLU rather than ReLU.** A `ReLU` neuron whose weighted sum stays negative has a zero derivative and never learns again. `LeakyReLU` keeps a small slope below zero, so it cannot die. The best configuration of the hyperoptimizer uses it.

**Cost.** `ReLU`-like functions need the He initialisation (see [Learning algorithm](#learning-algorithm)), and their outputs are not bounded, so a too high learning rate can make the `Sigmoid` output overflow.

## Output activation and output classes

**Choice.** 5 outputs, one per label: `Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black`, with a `Sigmoid` output layer. The prediction is the label of the highest output.

**Why.** One output per label needs no rule after the network: the answer is always exactly one label. Predicting the side that gives the check is the most complete answer, and the less detailed ones (`Check`, `Checkmate`) can be deduced from it. `Sigmoid` keeps each output in `[0, 1]`, the range of the one-hot targets.

**Alternatives.**

| Alternative | Why not |
|-------------|---------|
| 3 outputs (`Nothing`, `Check`, `Checkmate`), as in our first versions | Less information. We started there because it is an easier problem, then moved to 5 once the encoding worked. |
| `Linear` output | Needed when the target is not in `[0, 1]` (the Hamming distance of the bonus, see [BENCHMARKS.md](BENCHMARKS.md#what-we-learn)). For one-hot targets, it lets outputs drift outside the range for no benefit. |
| Softmax with cross-entropy | The standard for exclusive classes: real probabilities and stronger gradients. It needs a new output layer and a new loss in the module; `Sigmoid` with mean squared error already reaches about 96 %. |
| Stalemate as a 6th output | The dataset has no stalemate positions, so the network has nothing to learn it from. |

## Input encoding

**Choice.** King-Rays-Escape: 141 inputs that describe the board as seen from the king of the side to move (the first piece on each of the 8 rays, the knights and pawns that could attack, the 8 squares around the king and whether they are attacked).

**Evidence.** Five encodings were trained with the same network, the same data and the same seed ([BENCHMARKS.md](BENCHMARKS.md#input-formats), [INPUT_FORMATS_COMPARAISON.md](INPUT_FORMATS_COMPARAISON.md)):

| Encoding | Inputs | Accuracy of the best network |
|----------|-------:|-----------------------------:|
| Board-FEN (the raw FEN, one-hot) | 783 | ≈ 62 % |
| Board-Turn | 769 | ≈ 60 % |
| Dual-View | 1551 | ≈ 66 % |
| King-Rays | 133 | ≈ 80 % |
| **King-Rays-Escape** | **141** | **≈ 96 %** |

**Why.** A check is a relation between the king and a piece that can be on the other side of the board. With the raw board, the network has to learn that relation separately for every pair of squares, which needs far more positions and training time than we had. The king-relative encoding gives it that relation directly, with 5.5 times fewer inputs, so each training step is also faster. The squares around the king are what separates a check from a checkmate: adding them took the accuracy from 80 % to 96 %.

**Cost.** The encoding contains some chess knowledge (which squares are attacked), and it only fits questions about the king of the side to move. It does not give the answer: the network still learns, from labelled examples, which combinations of rays and attacked squares make a check or a checkmate.

## Size of the network and the other hyperparameters

**Choice.** The hyperparameters are found by `my_torch_hyperoptimizer`, which searches the layers, the learning rate, the momentum, the hidden activation, the initial bias and the batch size (random search, then successive halving, see [DESIGN_CHOICES.md](DESIGN_CHOICES.md#random-search-and-successive-halving)). Its result, `hyperopt/best.yml`, goes through the generator like any configuration. The curves of the search are in [BENCHMARKS.md](BENCHMARKS.md#hyperparameter-search).

**Evidence.** The best configuration found is `141-256-64-5`, `LeakyReLU` / `Sigmoid`, learning rate 0.0115, momentum 0.80, batch size 1. It went from 77.6 % after 5,000 positions to 90.9 % after 15,000 and 94.2 % after 45,000, and stayed first at every round.

**Alternatives.**

| Alternative | Why not |
|-------------|---------|
| Choosing by hand | Each try is a training of several minutes; a few tries cannot cover 7 hyperparameters. |
| Grid search | 3 values per hyperparameter is already 3⁷ = 2,187 trainings. |
| Genetic algorithm or other metaheuristics | Needs many full evaluations per generation; successive halving tries more configurations for the same budget, because the bad ones stop after 5,000 positions. |

**What the search tells us.**

- **Batch size.** None of the 7 configurations with `batchSize: 16` reaches the second round: with 16 times fewer updates for the same number of positions, they had not learnt enough in the short budget. Batch size 1 wins.
- **Momentum.** The 3 best configurations all have a momentum between 0.65 and 0.91.

## Learning algorithm

| Choice | Why |
|--------|-----|
| Backpropagation with gradient descent on labelled positions | The dataset gives the expected label of every position, so the error of each output is known. |
| Online updates, optional mini-batches (`batchSize`) | Online is the default because the hyperoptimizer found it best on chess; it also lets a training be stopped at any position with a usable network. Mini-batches are available for other tasks. |
| Momentum (`momentum`) | Speeds up the descent along stable directions; all the best configurations use it. |
| He initialisation for ReLU-like layers, Xavier for the others | With a fixed range, neurons with many inputs started saturated, and deep ReLU networks made the output overflow on the first sample (see [DESIGN_CHOICES.md](DESIGN_CHOICES.md#weight-initialisation)). |

## Avoiding over-fitting

| Choice | How it limits over-fitting | Setting |
|--------|----------------------------|---------|
| Validation set | Up to 500 positions (10 % of the file at most) are never trained on, and measure how the network does on unseen positions. | Enabled by `patience` |
| Early stopping | The validation accuracy is measured every `validationInterval` positions; after `patience` validations without progress, the analyzer offers to stop. | `patience`, `validationInterval` in the `.yml` |
| Keep the best network | `best.nn` is the network with the best validation accuracy; it is returned if the last one is worse. | Automatic with validation |
| Balanced epochs with a cap | Rare labels are drawn again, at most 4 times per position, so the network does not learn the rare positions by heart. | `MAX_OVERSAMPLING` |
| Shuffle at every epoch | The network cannot learn the order of the file. | Seed of the network |

**Why offer to stop rather than stop.** A validation set of 500 positions is noisy: the accuracy can stay flat for a while and then improve. The user sees the curve and decides; without a terminal, the training goes on and the best network is still kept.

## Network and configuration files

| Choice | Why | Alternative |
|--------|-----|-------------|
| `.nn` = the `MLP` object saved with `pickle` | Save and load in one call; every setting of the network is saved with it, so a `.nn` file loads without its configuration file, and any new attribute is saved automatically. | JSON with the layout and the weights: portable and safe, but a parser to maintain. See its risks in [CHOICE_IMPACT.md](CHOICE_IMPACT.md#loading-a-nn-file-from-an-unknown-source). |
| YAML configuration, checked key by key | Readable and writable by hand, with lists for the layers; `safe_load` cannot run code; a missing key or an unknown activation gives a clear message. 13 settings, only 4 required. | JSON (no comments, harder to write by hand), command-line options (too many). |
| A generator (`my_torch_generator config nb`) | Several networks from one configuration, to keep the best start; the hyperoptimizer writes a `.yml` that goes through it. | Generating in code only: still possible, since `MLP` is importable. |

## Reproducibility

| Choice | What it makes reproducible |
|--------|---------------------------|
| `seed` saved in every `.nn` | The initial weights, the shuffle of each epoch and the validation set: the same network trained on the same file twice gives the same result. |
| Same seed for every configuration of the hyperoptimizer | The differences of score come from the hyperparameters, not from a lucky initialisation. |
| `dataset.zip` committed, `dataset.sh` to build training files | The dataset is in the repository, with the script that builds the training files from it. |
| Checkpoints and `benchmarks/` scripts | Every curve of the documentation can be drawn again from a training. |

## Input validation and errors

**Choice.** Each line is checked with `python-chess` (6 FEN fields, legal position, known label); any error is printed on the error output and the program exits with `84`.

**Why.** A wrong label in a training file silently teaches the network something false. Checking legality by hand (two kings, no pawn on the last rank, side not to move not in check...) is long and error-prone, and `python-chess` is a well-tested reference. The network never sees it: its inputs are computed from the FEN string only.

## Link with the constraints of the project

How the choices above answer the constraints of the specification.

| Constraint of the project | Choices that answer it |
|---------------------------|------------------------|
| Language: anything that works on the dump, built with a `Makefile` (`re`, `clean`, `fclean`) | Python 3, nothing to compile; the `Makefile` links the four entry points. |
| A neural network module, dynamically loaded or directly imported, that creates, saves/loads and trains a network | The `src/my_torch/` package, imported by every program: `MLP(...)`, `save`, `load`, `learn`. |
| Libraries handling neural networks are not allowed | The network is plain Python; the libraries only do input/output, type checking and charts. |
| A machine-learning solution trained with supervised learning | Backpropagation on labelled positions; the label is learnt, never computed by the analyzer. |
| The outputs `Checkmate`, `Check`, `Nothing`, and the "full" level with the colour | 5 output classes, one per label. |
| A pre-trained `my_torch_network*` evaluated by the autograder | Pickle saves the network with all its settings, so it loads alone; prediction takes about 8 ms per position. |
| Justify all design choices, including the hyperparameters, with benchmarks and visual proof | This document, [DESIGN_CHOICES.md](DESIGN_CHOICES.md), [BENCHMARKS.md](BENCHMARKS.md) with charts drawn by `matplotlib`. |
| The training must try to avoid over-fitting | Validation set, early stopping, best network kept, capped oversampling, shuffle at every epoch. |
| Hyperparameter optimisation and optimisation of the learning (mini-batches, weight initialisation...) | The hyperoptimizer; mini-batches, momentum, He and Xavier initialisation. |
| Free file formats, "the more settings, the better" | YAML with 13 settings, pickle for the networks. |
| Keep every script and dataset to train the network again | Seeds saved in the networks, `dataset.zip`, `dataset.sh`, benchmark scripts. |
| Errors on the error output, exit code `84` | Validation of every line with `python-chess`, errors caught at the top of each program. |
