from __future__ import annotations

import argparse
import json
import os
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


@torch.no_grad()
def evaluate(model, loader, device, head_mask=None):
    model.eval()
    losses = []
    correct = []
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
        correct.append((logits.argmax(-1) == labels).float().cpu())

    losses = torch.cat(losses)
    correct = torch.cat(correct)
    return losses, correct


def main():
    parser = argparse.ArgumentParser(
        description="Calibration-seed stability experiment for QFC."
    )
    parser.add_argument("--model-id", default="textattack/bert-base-uncased-SST-2")
    parser.add_argument(
        "--model-revision",
        default="205ffbd1bc5c5b89802266f4948a601f53556b00",
    )
    parser.add_argument("--calibration-size", type=int, default=64)
    parser.add_argument("--evaluation-size", type=int, default=128)
    parser.add_argument("--heads-to-keep", type=int, default=6)
    parser.add_argument("--seeds", default="7,42,77")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--random-masks", type=int, default=30)
    parser.add_argument("--output-dir", default="results/step12_qfc_stability")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")

    train = load_dataset("stanfordnlp/sst2", split="train")
    validation = load_dataset("stanfordnlp/sst2", split="validation")
    if args.calibration_size > len(train):
        raise ValueError("calibration-size exceeds training split")
    if args.evaluation_size > len(validation):
        raise ValueError("evaluation-size exceeds validation split")

    evaluation = validation.select(range(args.evaluation_size))
    results = []

    for seed in seeds:
        calibration = train.shuffle(seed=seed).select(range(args.calibration_size))

        model, tokenizer = load_sequence_classifier(
            args.model_id,
            args.device,
            revision=args.model_revision,
        )

        cal_loader = make_text_loader(
            calibration,
            tokenizer,
            text_fields=("sentence",),
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        eval_loader = make_text_loader(
            evaluation,
            tokenizer,
            text_fields=("sentence",),
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        baseline_losses, baseline_correct = evaluate(
            model, eval_loader, args.device
        )

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
            **entropy_baseline_selections(vn, shannon, args.heads_to_keep),
        }
        validate_all(selections, layers, heads, args.heads_to_keep)

        seed_result = {
            "seed": seed,
            "baseline": {
                "accuracy": float(baseline_correct.mean()),
                "loss": float(baseline_losses.mean()),
            },
            "methods": {},
        }

        for name, selection in selections.items():
            mask = make_mask(
                selection, layers, heads, args.device
            )
            losses, correct = evaluate(
                model, eval_loader, args.device, mask
            )
            seed_result["methods"][name] = {
                "accuracy": float(correct.mean()),
                "loss": float(losses.mean()),
                "accuracy_delta": paired_bootstrap_delta(
                    correct,
                    baseline_correct,
                    n_boot=args.bootstrap,
                    seed=seed + 10000,
                ),
                "loss_delta": paired_bootstrap_delta(
                    losses,
                    baseline_losses,
                    n_boot=args.bootstrap,
                    seed=seed + 11000,
                ),
                "selected_heads_zero_based": selection,
            }

        # Random is a distribution over independent masks (seed = base + i).
        seed_result["methods"]["Random"] = evaluate_random_distribution(
            lambda m: (*evaluate(model, eval_loader, args.device, m), None, None),
            layers,
            heads,
            args.heads_to_keep,
            n_masks=args.random_masks,
            base_seed=seed + 1000,
            device=args.device,
            baseline=seed_result["baseline"],
        )

        results.append(seed_result)
        print(
            f"seed={seed} "
            + " ".join(
                f"{name}={entry['accuracy']:.4f}"
                for name, entry in seed_result["methods"].items()
            )
        )

    summary = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "dataset": "stanfordnlp/sst2",
        "calibration_split": "train",
        "evaluation_split": "validation_fixed_slice",
        "calibration_size": args.calibration_size,
        "evaluation_size": args.evaluation_size,
        "heads_to_keep_per_layer": args.heads_to_keep,
        "seeds": seeds,
        "runs": results,
        "aggregate": {},
        "research_note": (
            "Stability study. The evaluation set is fixed across calibration "
            "seeds; only calibration sampling changes. Random is the mean over "
            "random_masks independent masks (see Random.distribution); entropy "
            "baselines are reported keep-high and keep-low."
        ),
        "random_masks": args.random_masks,
    }

    method_names = list(results[0]["methods"])
    for name in method_names:
        accs = [r["methods"][name]["accuracy"] for r in results]
        losses = [r["methods"][name]["loss"] for r in results]
        summary["aggregate"][name] = {
            "accuracy_mean": mean(accs),
            "accuracy_std": pstdev(accs) if len(accs) > 1 else 0.0,
            "loss_mean": mean(losses),
            "loss_std": pstdev(losses) if len(losses) > 1 else 0.0,
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
