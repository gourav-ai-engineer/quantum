# Experiment Protocol — v0.1

## Step 2 objective

Validate that the new QFC implementation can:

1. extract attention matrices from a real fine-tuned Transformer;
2. convert them into density operators;
3. aggregate states without violating trace/PSD constraints;
4. compute pairwise quantum fidelity;
5. select exactly K heads per layer by greedy coverage;
6. physically prune the remaining heads; and
7. re-evaluate the same held-out examples.

## First run

Use BERT-base fine-tuned on SST-2 with:

- model: textattack/bert-base-uncased-SST-2
- dataset: stanfordnlp/sst2 validation
- first debug calibration: 256 examples
- final calibration: all 872 validation examples
- max length: 128
- 16 samples/batch for T4 initially
- 6 heads retained per layer for the 50% structured-pruning stress test

The public SST-2 dataset currently lists 872 validation examples and 1,821 test examples. The final paper should report the full validation split, not the 300-example subset used in the old manuscript.

## Acceptance criteria for this step

A run is considered technically successful only when:

- density matrices pass trace and PSD checks;
- fidelity values remain in [0, 1] up to numerical tolerance;
- every layer returns exactly K unique retained head indices;
- structured pruning changes parameter count;
- the pruned model executes without shape errors;
- baseline and pruned metrics are computed on exactly the same examples.

Do not record an accuracy improvement in the paper unless the experiment is rerun independently and the gain survives repeated evaluation.
