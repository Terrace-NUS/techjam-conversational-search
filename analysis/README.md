# Analysis workspace

This directory contains reproducible development analysis for the starter agent.

```text
build_expanded_eval_set.py  rebuilds the expanded dataset, priors, and reports
generated/                  generated evaluation datasets
reports/                    human-readable reports and machine-readable summaries
results/                    named evaluator runs used for comparisons
```

Run commands from the repository root so that the documented relative paths and
Python imports resolve consistently. The evaluator's default `results.json` stays
at the repository root as a disposable local output; named experiment results
belong in `analysis/results/`.

The main analysis write-up is
[`reports/optimization_results.md`](reports/optimization_results.md).
