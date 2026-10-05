# Results convention

Every number that goes into `docs/PROJECT_STATE.md` or the paper comes from a JSON in this tree.

```
results/<experiment>/<commit>/
    <task>_seed<N>.json  or  summary.json      # produced by the experiment script
    aggregate.json                              # v10 only (scripts/aggregate_v10_confirmatory.py)
    records.json.gz                             # v11 only: per-subset records ({task: {layer: [...]}}),
                                                #   kept out of summary.json so it stays small
    run_meta.json                               # commit, date (UTC), args, library versions, GPU
    log_<experiment>_<commit>_<YYYYMMDD>.txt    # console log
```

Experiments (directory names): `v10_confirmatory` (step18), `v6_budget_response` (step14),
`v5_mrpc_stability` (step13), `v11_objective_alignment` (step19). `<commit>` is the 12-character
short hash that was checked out when the run started; a rerun on the same commit needs `--tag`.

- Produce it with `scripts/run_gpu.sh` (or `colab/run_experiments.ipynb`); tiny `--smoke` runs go to
  `<results-root>/_smoke/` and are never results and never committed.
- Commit **small files only** (the runner warns above 5 MB). Large artifacts stay on Drive/Releases.
- Turn a directory into the ledger row: `python scripts/make_ledger_row.py results/<experiment>/<commit>`.
  It refuses smoke runs, runs recorded with other library versions (head_mask unreliable) and CPU runs.
- Never edit a committed JSON by hand. A wrong run is superseded by a new commit directory, and the old
  ledger row is marked superseded with the reason.
