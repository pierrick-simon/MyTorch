# Contributing

This document describes how we work on MyTorch: setup, code style, commits and continuous integration.

## Setup

```sh
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make                         # links the four executables to the root
./dataset.sh 2000            # only for the analyzer: extracts dataset.zip if needed, writes train.txt
```

The executables (`my_torch_generator`, `my_torch_analyzer`, `my_torch_bonus`, `my_torch_hyperoptimizer`) are hard links to the `entry_point` of their package in `src/`. They use `.venv/bin/python` as interpreter, with a path relative to the current directory: run them from the root of the repository. A new program needs its own `entry_point`, a rule in the `Makefile` (added to `all` and `fclean`) and a line in `.gitignore`.

## Code style

- **Linter**: [Ruff](https://docs.astral.sh/ruff/), run on `src/` by the CI. Before pushing:

  ```sh
  pip install ruff
  ruff check src
  ```

- **Naming**: `camelCase` for functions, methods and variables (`nbInputs`, `forceOverwrite`), `UPPER_CASE` for module constants (`EPOCHS`, `SEARCH_SPACE`), `PascalCase` for classes. A few older files (`fen_parser.py`, `parse_yml.py`) use `snake_case`: keep the style of the file you edit.
- **Type hints**: every function is annotated and decorated with `@typechecked` from `typeguard`. New functions should follow the same rule.
- **Imports**: absolute from the root of the repository (`from src.my_torch.MLP import MLP`), sorted as Ruff expects.
- **Module names**: do not rename or move the modules of `src/my_torch/`. `.nn` files are pickled objects and store their classes by module path, so a rename breaks every existing network (see [docs/DESIGN_CHOICES.md](docs/DESIGN_CHOICES.md#pickle-for-the-nn-format)).

## Common changes

| To add...                | Do this                                                                                                  |
|--------------------------|----------------------------------------------------------------------------------------------------------|
| An activation function   | A file in `src/my_torch/activationClasses/` with static `activation(sum, bias)` and `derivation(output)`, then an entry in `ActivationClassNames` and in the `ActivationClass` type of `__init__.py`. Document it in the README. |
| A configuration key      | An entry `"key": (type, required)` in `conf` of `src/my_torch_generator/parse_yml.py`, then use it in `generateNeuralNetwork`. Document it in the README table. |
| A bonus task             | A file in `src/bonus/tables/` with a `TABLE`, a `train(p, n)` and an `evaluate(p, n)`. Add it to `src/bonus/help.txt` and the README. |
| An analyzer label        | An entry in `STATE_LABELS` and in the `State` enum of `src/my_torch_analyzer/chessboard.py`, and one more value in `self.state` of `Chessboard.__init__`. Networks then need one more output (`nbOutputs`); the hyperoptimizer follows `len(STATE_LABELS)` on its own. |
| An analyzer input        | A feature function in `src/my_torch_analyzer/chessboard.py`, added to `_kingRelativeInput`, and the new size in `NB_INPUTS`. Every existing network and configuration needs the new `nbInputs`. Document it in [docs/CHESS_ANALYZER.md](docs/CHESS_ANALYZER.md#input-encoding). |
| A hyperparameter to search | An entry in `SEARCH_SPACE` of `src/my_torch_hyperoptimizer/search_space.py` (`choice`, `uniform` or `loguniform`), named like the key of the configuration file. Pass it to `MLP` in `buildNetwork` of `network.py`. `results.csv` and `best.yml` follow `SEARCH_SPACE` on their own. |
| A setting of the search  | A constant in `src/my_torch_hyperoptimizer/constants.py`: the program has no command line. |

## Commits

Commit messages follow the pattern `type(scope): description`, in lowercase:

```
feat(nn): add of multi-activation-function multi-layer-perceptron
fix(analyzer): state can be an str or none
feat(doc): readme
```

| Type    | Use                                       |
|---------|-------------------------------------------|
| `feat`  | new feature                               |
| `fix`   | bug fix, lint fix                         |
| `docs`  | documentation only                        |
| `merge` | merge commit                              |

Common scopes: `nn` (the library and the generator), `analyzer`, `bonus`, `hyperoptimizer`, `readme`, `benchmarks`, `doc`, `ruff`, `compilation`, `init`.

## Continuous integration

Two GitHub Actions workflows run on every push and pull request:

| Workflow                                         | What it does                                                                                       |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------|
| [`ruff.yml`](.github/workflows/ruff.yml)         | `ruff check` on `src/`. Also runs on its own on pull requests that change Python files.            |
| [`mirror.yml`](.github/workflows/mirror.yml)     | Runs `make` (5 minutes at most), checks that `my_torch_analyzer` and `my_torch_generator` exist and are executable, runs Ruff, then pushes the repository to the Epitech mirror. |

The mirror only receives commits that pass both checks. Branches whose name starts with `ga-ignore-` are skipped by the workflows.

## Benchmarks

Scripts in [`benchmarks/`](benchmarks/) produce the tables and charts of [docs/BENCHMARKS.md](docs/BENCHMARKS.md) (each script is described in its [Setup](docs/BENCHMARKS.md#setup) section). After a change to the library, run them again and update the document:

```sh
.venv/bin/python -O benchmarks/speed.py
.venv/bin/python -O benchmarks/tables.py
.venv/bin/python benchmarks/plots.py         # copy the new numbers into plots.py first
```

`benchmarks/chess_dataset.py` was written for the former 783-input encoding and no longer runs. To measure a chess network, train it and evaluate its checkpoints:

```sh
.venv/bin/python -O my_torch_analyzer --train --save chess_trained.nn chess_1.nn train.txt
.venv/bin/python -O benchmarks/checkpoints_csv.py chess_trained test.txt
.venv/bin/python benchmarks/checkpoints_plot.py chess_trained/checkpoints.csv -o chess_trained.png
```

`checkpoints_plot.py` accepts several CSV files, one curve each, to compare trainings.
