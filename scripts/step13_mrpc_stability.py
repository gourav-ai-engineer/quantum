from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from statistics import mean, pstdev

import torch
from datasets import load_dataset

from qfc.baselines import (
    entropy_baseline_selections,
    evaluate_random_distribution,
    validate_all,
)
from qfc.coverage import greedy_select, weighted_greedy_select
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import (
    _move_batch,
    collect_mean_density_states,
    load_sequence_classifier,
    make_text_loader,
    michel_head_importance,
)
from qfc.metrics import paired_bootstrap_delta


def shannon_scores(model, loader, device):
    model.eval()
    sums = None
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            batch.pop("labels", None)
            mask = batch["attention_mask"].float()
            out = model(**batch, output_attentions=True, return_dict=True)
            if out.attentions is None:
                raise RuntimeError("attention outputs unavailable")
            if sums is None:
                sums = [torch.zeros(a.shape[1], dtype=torch.float64) for a in out.attentions]

            valid_queries = mask.sum(-1).clamp_min(1.0)
            for li, attn in enumerate(out.attentions):
                p = attn.float().clamp_min(1e-12)
                p = p * mask[:, None, None, :]
                entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
                entropy = entropy * mask[:, None, :]
                per_example = entropy.sum(-1) / valid_queries[:, None]
                sums[li] += per_example.sum(0).cpu().double()
            count += int(mask.shape[0])

    if sums is None or count == 0:
        raise ValueError("empty loader")
    return [s / count for s in sums]


def vn_scores(mean_states):
    result = []
    for states in mean_states:
        vals = []
        for rho in states:
            eig = torch.linalg.eigvalsh(0.5 * (rho + rho.T)).clamp_min(1e-12)
            vals.append(-(eig * eig.log()).sum())
        result.append(torch.stack(vals))
    return result


def select_topk(scores, k):
    return {
        layer: torch.topk(score, k=k, largest=True).indices.tolist()
        for layer, score in enumerate(scores)
    }


def random_select(layers, heads, k, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        layer: torch.randperm(heads, generator=g)[:k].tolist()
        for layer in range(layers)
    }


def make_mask(selection, layers, heads, device):
    mask = torch.zeros((layers, heads), dtype=torch.float32, device=device)
    for layer, hs in selection.items():
        mask[layer, hs] = 1.0
    return mask


def selection_jaccard(selection_a, selection_b):
    """Jaccard overlap over all (layer, head) selections."""
    set_a = {
        (layer, head)
        for layer, heads in selection_a.items()
        for head in heads
    }
    set_b = {
        (layer, head)
        for layer, heads in selection_b.items()
        for head in heads
    }
    union = set_a | set_b
    if not union:
        raise ValueError("cannot compute Jaccard overlap for empty selections")
    return float(len(set_a & set_b) / len(union))


def selection_stability(results, method_name):
    """Pairwise selection Jaccard across calibration seeds."""
    selections = [
        run["methods"][method_name]["selected_heads_zero_based"]
        for run in results
    ]
    values = [
        selection_jaccard(a, b)
        for a, b in combinations(selections, 2)
    ]
    return {
        "mean": mean(values) if values else 1.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "pairwise": values,
    }



def f1_score(labels, predictions):
    labels = labels.long()
    predictions = predictions.long()
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


def bootstrap_f1_delta(
    method_labels,
    method_predictions,
    baseline_labels,
    baseline_predictions,
    n_boot,
    seed,
    confidence=0.95,
):
    if not (
        len(method_labels)
        == len(method_predictions)
        == len(baseline_labels)
        == len(baseline_predictions)
    ):
        raise ValueError("paired arrays must have equal length")

    n = len(baseline_labels)
    if n == 0:
        raise ValueError("cannot bootstrap empty arrays")

    g = torch.Generator().manual_seed(seed)
    deltas = torch.empty(n_boot, dtype=torch.float64)
    for i in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        method_f1 = f1_score(method_labels[idx], method_predictions[idx])
        baseline_f1 = f1_score(
            baseline_labels[idx], baseline_predictions[idx]
        )
        deltas[i] = method_f1 - baseline_f1

    alpha = (1.0 - confidence) / 2.0
    lo = float(torch.quantile(deltas, alpha))
    hi = float(torch.quantile(deltas, 1.0 - alpha))
    return {"estimate": float(deltas.mean()), "ci_low": lo, "ci_high": hi}


@torch.no_grad()
def evaluate(model, loader, device, head_mask=None):
    model.eval()
    losses = []
    labels_all = []
    predictions_all = []

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        out = model(
            **batch,
            labels=labels,
            head_mask=head_mask,
            return_dict=True,
        )
        logits = out.logits.float()
        losses.append(
            torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            ).cpu()
        )
        labels_all.append(labels.detach().cpu())
        predictions_all.append(logits.argmax(-1).detach().cpu())

    losses = torch.cat(losses)
    labels_all = torch.cat(labels_all)
    predictions_all = torch.cat(predictions_all)
    correct = (predictions_all == labels_all).float()
    return losses, correct, labels_all, predictions_all


def main():
    parser = argparse.ArgumentParser(
        description="Cross-task calibration-seed stability study for QFC on MRPC."
    )
    parser.add_argument(
        "--model-id",
        default="textattack/bert-base-uncased-MRPC",
    )
    parser.add_argument(
        "--model-revision",
        default="ddeddf4a04cd7b9415b00e40b00e78f0c61a7921",
    )
    parser.add_argument("--calibration-size", type=int, default=128)
    parser.add_argument("--evaluation-size", type=int, default=-1)
    parser.add_argument("--heads-to-keep", type=int, default=6)
    parser.add_argument("--seeds", default="7,42,77")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--random-masks", type=int, default=30)
    parser.add_argument(
        "--output-dir", default="results/step13_mrpc_stability"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")

    train = load_dataset("glue", "mrpc", split="train")
    validation = load_dataset("glue", "mrpc", split="validation")
    if args.calibration_size > len(train):
        raise ValueError("calibration-size exceeds training split")

    if args.evaluation_size < 0:
        evaluation = validation
    else:
        if args.evaluation_size > len(validation):
            raise ValueError("evaluation-size exceeds validation split")
        evaluation = validation.select(range(args.evaluation_size))

    results = []

    for seed in seeds:
        calibration = train.shuffle(seed=seed).select(
            range(args.calibration_size)
        )

        model, tokenizer = load_sequence_classifier(
            args.model_id,
            args.device,
            revision=args.model_revision,
        )

        cal_loader = make_text_loader(
            calibration,
            tokenizer,
            text_fields=("sentence1", "sentence2"),
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        eval_loader = make_text_loader(
            evaluation,
            tokenizer,
            text_fields=("sentence1", "sentence2"),
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        (
            baseline_losses,
            baseline_correct,
            baseline_labels,
            baseline_predictions,
        ) = evaluate(model, eval_loader, args.device)

        states = collect_mean_density_states(
            model, cal_loader, args.device
        )
        shannon = shannon_scores(model, cal_loader, args.device)
        vn = vn_scores(states)
        michel = michel_head_importance(
            model, cal_loader, args.device
        ).detach().cpu()

        layers = len(states)
        heads = int(states[0].shape[0])

        qfc = {}
        iwqfc = {}
        for layer, layer_states in enumerate(states):
            sim = pairwise_fidelity(layer_states)
            qfc[layer], _ = greedy_select(sim, args.heads_to_keep)

            raw = michel[layer].clamp_min(0.0)
            if float(raw.sum()) > 0:
                weights = raw / raw.sum()
            else:
                weights = torch.full_like(
                    raw, 1.0 / len(raw)
                )
            iwqfc[layer], _ = weighted_greedy_select(
                sim, weights, args.heads_to_keep
            )

        selections = {
            "QFC": qfc,
            "IWQFC": iwqfc,
            "MichelGate": select_topk(
                [michel[i] for i in range(layers)],
                args.heads_to_keep,
            ),
            # Entropy baselines in both directions, labelled explicitly.
            **entropy_baseline_selections(vn, shannon, args.heads_to_keep),
        }
        validate_all(selections, layers, heads, args.heads_to_keep)

        seed_result = {
            "seed": seed,
            "baseline": {
                "accuracy": float(baseline_correct.mean()),
                "f1": f1_score(
                    baseline_labels, baseline_predictions
                ),
                "loss": float(baseline_losses.mean()),
            },
            "methods": {},
        }

        for name, selection in selections.items():
            mask = make_mask(
                selection, layers, heads, args.device
            )
            losses, correct, labels, predictions = evaluate(
                model, eval_loader, args.device, mask
            )
            seed_result["methods"][name] = {
                "accuracy": float(correct.mean()),
                "f1": f1_score(labels, predictions),
                "loss": float(losses.mean()),
                "accuracy_delta": paired_bootstrap_delta(
                    correct,
                    baseline_correct,
                    n_boot=args.bootstrap,
                    seed=seed + 10000,
                ),
                "f1_delta": bootstrap_f1_delta(
                    labels,
                    predictions,
                    baseline_labels,
                    baseline_predictions,
                    n_boot=args.bootstrap,
                    seed=seed + 12000,
                ),
                "loss_delta": paired_bootstrap_delta(
                    losses,
                    baseline_losses,
                    n_boot=args.bootstrap,
                    seed=seed + 11000,
                ),
                "selected_heads_zero_based": selection,
            }

        # Random is a distribution over independent masks, fixed across
        # calibration seeds because it does not use calibration data.
        seed_result["methods"]["Random"] = evaluate_random_distribution(
            lambda m: evaluate(model, eval_loader, args.device, m),
            layers,
            heads,
            args.heads_to_keep,
            n_masks=args.random_masks,
            base_seed=2027,
            device=args.device,
            with_f1=True,
            baseline=seed_result["baseline"],
        )

        results.append(seed_result)
        print(
            f"seed={seed} "
            + " | ".join(
                f"{name} acc={entry['accuracy']:.4f} F1={entry['f1']:.4f}"
                for name, entry in seed_result["methods"].items()
            )
        )

    summary = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "dataset": "glue/mrpc",
        "calibration_split": "train",
        "evaluation_split": (
            "validation_full" if args.evaluation_size < 0 else "validation_fixed_slice"
        ),
        "calibration_size": args.calibration_size,
        "evaluation_size": len(evaluation),
        "heads_to_keep_per_layer": args.heads_to_keep,
        "seeds": seeds,
        "runs": results,
        "aggregate": {},
        "research_note": (
            "Cross-task stability study. The evaluation set is fixed across "
            "calibration seeds. Random is the mean over random_masks independent "
            "masks (seed = 2027 + i), fixed across calibration seeds because it "
            "does not use calibration data; see methods.Random.distribution. "
            "Entropy baselines are reported keep-high and keep-low."
        ),
        "random_masks": args.random_masks,
    }

    method_names = list(results[0]["methods"])
    for name in method_names:
        accs = [r["methods"][name]["accuracy"] for r in results]
        f1s = [r["methods"][name]["f1"] for r in results]
        losses = [r["methods"][name]["loss"] for r in results]
        stability = selection_stability(results, name)
        summary["aggregate"][name] = {
            "accuracy_mean": mean(accs),
            "accuracy_std": pstdev(accs) if len(accs) > 1 else 0.0,
            "f1_mean": mean(f1s),
            "f1_std": pstdev(f1s) if len(f1s) > 1 else 0.0,
            "loss_mean": mean(losses),
            "loss_std": pstdev(losses) if len(losses) > 1 else 0.0,
            "selection_jaccard_mean": stability["mean"],
            "selection_jaccard_std": stability["std"],
            "selection_jaccard_pairwise": stability["pairwise"],
        }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
