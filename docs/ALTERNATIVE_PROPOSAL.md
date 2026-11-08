# Alternative proposal

If we had to start the project again, this is what we would do differently, and **the result each change gives**. Every result below was measured, except the one marked as an estimate.

- [How the results were measured](#how-the-results-were-measured)
- [1. Start with the king-relative encoding](#1-start-with-the-king-relative-encoding)
- [2. Search the hyperparameters from the start](#2-search-the-hyperparameters-from-the-start)
- [3. Train on much more data](#3-train-on-much-more-data)
- [4. Run the hyperoptimizer in parallel](#4-run-the-hyperoptimizer-in-parallel)
- [Summary](#summary)

## How the results were measured

Alternatives 1 and 2 come from our own trainings ([BENCHMARKS.md](BENCHMARKS.md), `hyperopt/results.csv`).

Alternative 3 needs many long trainings, so it was measured with a **prototype** of our best network written with NumPy ([`benchmarks/alternatives.py`](../benchmarks/alternatives.py)), which runs them in seconds. The prototype has the same shape (141-256-64-5), the same activations (LeakyReLU / Sigmoid), the same initialisation (He / Xavier), the same balanced epochs and the same hyperparameters as the best configuration of `my_torch_hyperoptimizer`. It is trained and validated on the same data as the hyperoptimizer: 7,200 training positions and 1,800 validation positions.

To check that the prototype learns like MyTorch, we compared its validation accuracy (mean of 5 seeds) with the one MyTorch reached in the hyperoptimizer:

| Positions seen | MyTorch (hyperoptimizer) | Prototype |
|---------------:|-------------------------:|----------:|
| 5,000          | 77.6 %                   | 77.5 %    |
| 15,000         | 90.9 %                   | 91.5 %    |
| 45,000         | 94.2 %                   | 94.4 %    |

The two learn the same way, so the results measured with the prototype hold for MyTorch. The accuracies below are the mean of 3 seeds on the 1,800 validation positions.

```sh
OMP_NUM_THREADS=1 .venv/bin/python -O benchmarks/alternatives.py data
```

## 1. Start with the king-relative encoding

**What we did.** Our first encoding was the raw board: 783 inputs (64 squares × 12 pieces, plus the other FEN fields). We spent a large part of the project tuning networks on it before changing the encoding.

**What we would do.** Think first about what decides the answer, a relation between the king and the pieces that attack it, and give the network that relation from the start.

**Result** (same network, same data, same seed, [BENCHMARKS.md](BENCHMARKS.md#input-formats)):

| Encoding | Inputs | Accuracy |
|----------|-------:|---------:|
| Raw board (Board-FEN) | 783 | ≈ 62 % |
| King-Rays-Escape | 141 | **≈ 96 %** |

**+34 points**, with 5.5 times fewer inputs, so each training step is also faster. None of the tuning done on the raw board got close: the best raw-board result was 68.8 %.

## 2. Search the hyperparameters from the start

**What we did.** The hyperparameters were chosen by hand until late in the project.

**What we would do.** Write the hyperoptimizer first, and let it choose.

**Result.** In the hyperoptimizer, 27 configurations were drawn at random in a reasonable range (`hyperopt/results.csv`). After 5,000 positions:

| Configuration | Validation accuracy |
|---------------|--------------------:|
| Median of the 27 | 44.8 % |
| Best one | **77.6 %** |

A configuration chosen without a search is, on average, the median one: the search gives **+33 points** for the same training budget. The best configuration then reached 94.2 % after 45,000 positions. The search also answers questions that would otherwise take many manual tries: the 7 `Sigmoid` hidden layers and the 7 configurations with `batchSize: 16` were all eliminated in the first round.

## 3. Train on much more data

**What we did.** The trainings and the hyperoptimizer use a few thousand positions. The dataset holds about 720,000.

**What we would do.** Train the final network on ten times more positions.

**Result.** 72,000 training positions (8,000 from each of the 9 files of the dataset, none of them in the validation set), against the 7,200 of the hyperoptimizer, same network, batches of 32:

| Training positions | Positions seen | Validation accuracy |
|-------------------:|---------------:|--------------------:|
| 7,200 | 900,000 | 94.7 % |
| 72,000 | 900,000 | 96.7 % |
| 72,000 | 3,000,000 | **97.1 %** |

With 7,200 positions, the accuracy stops around 94 to 95 % however long the training: the limit is the data, not the network. Ten times more data gives **+2.4 points**.

## 4. Run the hyperoptimizer in parallel

**What we did.** The hyperoptimizer trains its configurations one after the other, on one core.

**What we would do.** Train the configurations of a round in parallel, one process per configuration (`multiprocessing`). They are independent, so no synchronisation is needed inside a round.

**Result (estimate).** One run trains 27 configurations on 5,000 positions, then 9 on 10,000 more, then 3 on 30,000 more: 315,000 positions in a row. On our 16-core machine, the first round runs in 2 waves of 16, and the next two in one wave each: the time of 50,000 positions. That is about **6 times faster**, or 6 times more configurations tried in the same time. The real gain is a little lower, since the configurations do not all have the same size and the slowest one sets the time of a round.

## Summary

| Alternative | Current result | Result with the alternative | Evidence |
|-------------|----------------|-----------------------------|----------|
| King-relative encoding from the start | ≈ 62 % (raw board) | ≈ 96 % | Measured |
| Hyperparameter search from the start | 44.8 % (median configuration) | 77.6 % (best), same budget | Measured |
| Ten times more training data | ≈ 94–95 % | 97.1 % | Measured |
| Parallel hyperoptimizer | 1 core | ≈ 6 × faster on 16 cores | Estimate |

If we started again, we would combine 1 to 4: the king-relative encoding from day one, hyperparameters chosen by a parallel search, and a final network trained on ten times more data. Together, they give a better network (≈ 97 % instead of ≈ 94 % on the same validation set).
