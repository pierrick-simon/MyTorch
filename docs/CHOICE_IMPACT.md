# Choice impact assessment

This document looks at the consequences of the technical choices of MyTorch: their legal and ethical limits, their cost for a user or a business, how a user can misuse them, and how the programs guide the user. Numbers come from [BENCHMARKS.md](BENCHMARKS.md) unless stated otherwise.

- [Legal and ethical limits](#legal-and-ethical-limits)
- [Technical choices and their business impact](#technical-choices-and-their-business-impact)
- [How a technical choice can be misused](#how-a-technical-choice-can-be-misused)
- [User experience](#user-experience)
- [Summary](#summary)

## Legal and ethical limits

**Licence of the dependencies.** The analyzer depends on [`python-chess`](https://python-chess.readthedocs.io/), which is licensed under **GPL-3.0 or later**. Any version of MyTorch distributed with it must be distributed under a compatible licence, with its source code. The other dependencies are permissive: `typeguard` (MIT), `click` (BSD-3-Clause), `tqdm` (MPL-2.0 and MIT), `matplotlib` (PSF-based), `PyYAML` (MIT). The repository has no `LICENSE` file yet, so by default nobody else is allowed to reuse the code. A licence has to be chosen before any public release, and it must be compatible with the GPL.

**Dataset.** The training positions come from `dataset.zip`, provided with the project. They contain chess positions only, with no player names, ratings or game history, so the network processes no personal data. They may not be redistributed beyond what their provider allows.

**What the network must not be used for.**

- **Arbitration.** The network is right about 96 % of the time (967 positions out of 1,000 of `test.txt` for `my_torch_network.nn`). It must never decide the result of a real game: a rules engine like `python-chess` is exact, the network is not. The project exists to show that a network can *learn* the rules, not to replace them.
- **Fair play.** Using any computer help during a rated game breaks the rules of every chess federation and online platform. The analyzer gives very little help, since it suggests no move and only says whether a king is in check or checkmated, but it still must not run during a game.
- **Explainability.** The network gives a label, not a reason. A user who needs to know *why* a position is a checkmate (teaching, refereeing) needs a rules engine, not MyTorch.

## Technical choices and their business impact

| Choice | Benefit | Cost | Business impact |
|--------|---------|------|-----------------|
| **Pure Python, no NumPy** | Every computation is readable and was written by us, as the subject requires. | Slow: one training step of a 141-64-32-5 network takes 1.17 ms with `-O`, so one pass over the dataset takes about 14 minutes on one core. | A training that a vectorised library would run in seconds takes hours. Each try costs developer time and machine time, so fewer configurations can be tested. |
| **King-relative input encoding (141 inputs)** | ≈ 96 % accuracy, against 60 to 66 % for raw-board encodings with the same network (see [INPUT_FORMATS_COMPARAISON.md](INPUT_FORMATS_COMPARAISON.md)). | It only works for this task: a new task needs a new encoding. | Six times fewer inputs than the raw board, so the network is smaller, faster to train and faster to run. |
| **Runtime type checking (`typeguard`)** | Type mistakes are reported where they happen. | Training is 9 to 16 times slower unless Python runs with `-O`. | Fewer bugs during development, but a user who forgets `-O` pays the full cost (see [misuse](#forgetting--o)). |
| **Pickle for the `.nn` format** | One line to save and load the whole network, with no format to maintain. | Not secure, Python only, and tied to the names of our classes. | Fast to develop, but a network cannot be shared safely or loaded by another language (see [misuse](#loading-a-nn-file-from-an-unknown-source)). |
| **Hyperoptimizer (random search + successive halving)** | Finds a configuration without testing each one by hand: 27 configurations, most of them stopped early. | Several hours of computation. | Replaces days of manual trial and error with one command that runs unattended. |

## How a technical choice can be misused

These are the cases where a user, without any bad intent, uses the programs in a way that gives a wrong result, loses work, or creates a security risk. For each one: what happens, why, what already protects the user, and what is still missing.

### Loading a `.nn` file from an unknown source

- **Choice:** `.nn` files are Python pickles ([MLP.py](../src/my_torch/MLP.py), `MLP.load`).
- **Misuse:** a user downloads a "pre-trained network" from someone else and runs `./my_torch_analyzer --predict their_network.nn boards.txt`.
- **Consequence:** loading a pickle can **run any code** written by the author of the file, with the user's rights: deleting files, stealing SSH keys, installing malware. Nothing in the program can detect it, because the code runs during the load, before the network is used.
- **Protection today:** none.
- **What should be done:** only load `.nn` files you created yourself, and say so in the README. A safe format (the weights and the configuration in JSON, or `numpy.save` without pickle) would remove the risk completely.

### Training without `--save`

- **Choice:** as the subject requires, `--train` without `--save` saves the trained network back into `LOADFILE`.
- **Misuse:** `./my_torch_analyzer --train my_torch_network.nn new_data.txt` to "try something".
- **Consequence:** the delivered network is overwritten **without confirmation**, even if the new training is worse or stopped with `Ctrl+C` after a few seconds.
- **Protection today:** the checkpoints of the training are kept in `my_torch_network/`, and git keeps the committed network.
- **What should be done:** always use `--save` when experimenting; the analyzer could warn when it is about to overwrite `LOADFILE`.

### Choosing a `--save` name that matches an existing directory

- **Choice:** checkpoints go into a directory named after `SAVEFILE` without `.nn`, and that directory is emptied at the start of each training ([train.py](../src/my_torch_analyzer/train.py), `prepareSaveDir`).
- **Misuse:** a user keeps their networks in `models/` and runs `--train --save models.nn ...`.
- **Consequence:** every `.nn` file in `models/` is **deleted** without confirmation.
- **Protection today:** the analyzer refuses to start if the directory contains anything other than `.nn` files, so a directory with configuration files or datasets is safe.
- **What should be done:** ask for confirmation before deleting, or create a new directory for each training (for example with the date).

### Reading the output of the network as a probability

- **Choice:** the output layer uses `Sigmoid` and the training is balanced by drawing rare labels again ([balanced epochs](DESIGN_CHOICES.md#balanced-epochs)).
- **Misuse:** a user reads an output of `0.8` as "80 % sure it is a checkmate".
- **Consequence:** the outputs are not calibrated probabilities. The balancing changes how often each label was seen, so the outputs do not reflect how frequent each state is in real games. A decision based on a threshold would be wrong.
- **Protection today:** `--predict` only prints the label with the highest output, never the raw values.
- **What should be done:** document it; a softmax output layer trained with cross-entropy would give real probabilities.

### Measuring the accuracy on a test set that overlaps the training set

- **Choice:** `dataset.sh` picks positions at random without a seed.
- **Misuse:** `./dataset.sh 2000 train.txt` then `./dataset.sh 300 test.txt`, and the test accuracy is reported as the accuracy of the network.
- **Consequence:** the two files can share positions, so the accuracy is **too optimistic**: the network is partly tested on positions it learned by heart. The same happens if the hyperoptimizer's validation score is reported as a final score, because that set was used to choose the configuration.
- **Protection today:** the README and [CHESS_ANALYZER.md](CHESS_ANALYZER.md) warn about it and show how to split one shuffled file instead.
- **What should be done:** a seed for `dataset.sh`, and a test set that is never used during training or the search.

### Using the network outside the distribution it was trained on

- **Choice:** the network is trained on the 9 files of the dataset (3 states × 3 piece counts).
- **Misuse:** asking about positions that look nothing like these: openings with many pieces, puzzles, chess variants.
- **Consequence:** the accuracy measured on `test.txt` is not guaranteed. The network still gives an answer, with no warning.
- **Protection today:** none, apart from rejecting illegal positions.
- **What should be done:** document the training distribution, and measure the accuracy separately for each piece count.

### Forgetting `-O`

- **Choice:** every function is checked by `typeguard`, which `-O` disables.
- **Misuse:** launching a long training with `./my_torch_analyzer --train ...` instead of `.venv/bin/python -O my_torch_analyzer --train ...`.
- **Consequence:** the training runs 9 to 16 times slower: a training of one hour takes a day.
- **Protection today:** documented in the README ([Optimized mode](../README.md#optimized-mode--o)).
- **What should be done:** print a reminder at the start of a training when `-O` is not set (`__debug__` is `True`).

### Using too high a learning rate

- **Choice:** the learning rate is free in the configuration file, and `Sigmoid` uses `math.exp`.
- **Misuse:** a user raises the learning rate to train faster.
- **Consequence:** the weights explode and the analyzer stops with a Python `OverflowError` instead of a clear message. The checkpoints already saved are kept.
- **Protection today:** the hyperoptimizer catches the error and scores the configuration 0; the analyzer does not.
- **What should be done:** catch the error in the analyzer, explain that the learning rate is too high, and keep the last good network.

### Stopping a prediction with `Ctrl+C`

- **Choice:** `predict` catches `KeyboardInterrupt` to stop cleanly.
- **Misuse:** a script runs `--predict` and its output is used by another program.
- **Consequence:** an interrupted prediction exits with code `0`, so the caller cannot tell a partial output from a complete one.
- **What should be done:** exit with a non-zero code when the prediction is interrupted.

## User experience

The programs are designed so that the user always knows what is happening:

- **Progress.** Training shows a `tqdm` progress bar per epoch, with the elapsed time, the speed and the time left, and the validation accuracy next to it.
- **Feedback.** The analyzer prints the number of positions kept for validation, the accuracy before training, each validation, and at the end the number of iterations and whether the best or the last network was kept.
- **Early stopping on demand.** When the validation accuracy stops improving `patience` times in a row, the analyzer asks whether to stop and keep the best network, instead of stopping on its own. Without a terminal (in a script), it only prints a warning.
- **Nothing is lost.** `Ctrl+C` during training saves the network trained so far, and a checkpoint is saved every 1,000 positions.
- **Safe defaults.** The generator asks before overwriting a `.nn` file, the save warns when the extension is not `.nn`, and the analyzer refuses to empty a directory that contains files other than `.nn`.
- **Errors.** Every error is printed on the error output with the reason (the line, the field, the expected label), and the programs exit with code `84`. `--help` describes every option.
- **Readable output.** `--predict` prints the same labels as the training files (`Check White`, `Checkmate Black`...), so its output can be compared line by line with an expected file.

## Summary

| Area | Covered | Still to do |
|------|---------|-------------|
| Legal and ethical limits | Dataset without personal data, network not suitable for arbitration or during a game | Choose a licence compatible with the GPL of `python-chess` |
| Business impact | Speed of pure Python, gain of the input encoding, cost of type checking, hyperoptimizer | — |
| Misuse | Nine cases listed with their consequence | Safe `.nn` format, confirmation before overwriting or deleting, `OverflowError` in the analyzer, seeded `dataset.sh` |
| User experience | Progress bars, validation, early stopping on demand, checkpoints, clear errors | Reminder when `-O` is not set |
