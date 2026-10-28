# Contributing

This document describes how we work on MyTorch: setup, code style, commits and continuous integration.

## Setup

```sh
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
unzip dataset.zip            # only needed for the chess analyzer and its benchmark
```

The executables (`my_torch_generator`, `my_torch_analyzer`, `my_torch_bonus`) use `.venv/bin/python` as interpreter, with a path relative to the current directory: run them from the root of the repository.

## Code style

- **Linter**: [Ruff](https://docs.astral.sh/ruff/), run on `src/` by the CI. Before pushing:

  ```sh
  pip install ruff
  ruff check src
  ```

- **Naming**: `camelCase` for functions, methods and variables of the library and the generator (`nbInputs`, `forceOverwrite`), `PascalCase` for classes. Keep the style of the file you edit.
- **Type hints**: every function is annotated and decorated with `@typechecked` from `typeguard`. New functions should follow the same rule.
- **Imports**: absolute from the root of the repository (`from src.my_torch.MLP import MLP`), sorted as Ruff expects.
- **Module names**: do not rename or move the modules of `src/my_torch/`. `.nn` files are pickled objects and store their classes by module path, so a rename breaks every existing network (see [docs/DESIGN_CHOICES.md](docs/DESIGN_CHOICES.md#pickle-for-the-nn-format)).

## Common changes

| To add...                | Do this                                                                                                  |
|--------------------------|----------------------------------------------------------------------------------------------------------|
| An activation function   | A file in `src/my_torch/activationClasses/` with static `activation(sum, bias)` and `derivation(output)`, then an entry in `ActivationClassNames` and in the `ActivationClass` type of `__init__.py`. Document it in the README. |
| A configuration key      | An entry `"key": (type, required)` in `conf` of `src/my_torch_generator/parse_yml.py`, then use it in `generateNeuralNetwork`. Document it in the README table. |
| A bonus task             | A file in `src/bonus/tables/` with a `TABLE`, a `train(p, n)` and an `evaluate(p, n)`. Add it to `src/bonus/help.txt` and the README. |
| An analyzer label        | An entry in `STATE_LABELS` and, for a new class, in the `State` enum of `src/my_torch_analyzer/Chessboard.py`. Networks then need one more output. |

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
| `merge` | merge commit                              |

Common scopes: `nn` (the library and the generator), `analyzer`, `bonus`, `doc`, `ruff`, `init`.

## Continuous integration

Two GitHub Actions workflows run on every push and pull request:

| Workflow                                         | What it does                                                                                       |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------|
| [`ruff.yml`](.github/workflows/ruff.yml)         | `ruff check` on `src/`. Also runs on its own on pull requests that change Python files.            |
| [`mirror.yml`](.github/workflows/mirror.yml)     | Checks that `my_torch_analyzer` and `my_torch_generator` exist and are executable, runs Ruff, then pushes the repository to the Epitech mirror. |

The mirror only receives commits that pass both checks. Branches whose name starts with `ga-ignore-` are skipped by the workflows.

## Benchmarks

Scripts in [`benchmarks/`](benchmarks/) produce the tables of [docs/BENCHMARKS.md](docs/BENCHMARKS.md). After a change to the library, run them again and update the document:

```sh
.venv/bin/python -O benchmarks/speed.py
.venv/bin/python -O benchmarks/tables.py
.venv/bin/python -O benchmarks/chess_dataset.py
```
