from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Iterable

import torch
from datasets import load_dataset

from qfc.coverage import greedy_select
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import (
    _move_batch,
    collect_mean_density_states,
    load_sequence_classifier,
    make_text_loader,
    michel_head_importance,
    parameter_count,
    structured_prune,
)
from qfc.metrics import evaluate_per_example, paired_bootstrap_delta
from qfc.states import density_operator


def shannon_entropy_scores(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    device: str,
):
    model.eval()
    sums = None
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            batch.pop("labels", None)
            attention_mask = batch["attention_mask"].float()

            out = model(
                **batch,
                output_attentions=True,
                return_dict=True,
            )
            if out.attentions is None:
                raise RuntimeError("Model did not return attentions")

            if sums is None:
                sums = [
                    torch.zeros(a.shape[1], dtype=torch.float64)
                    for a in out.attentions
                ]

            valid_queries = attention_mask.sum(-1).clamp_min(1.0)
            for li, attn in enumerate(out.attentions):
                p = attn.float().clamp_min(1e-12)
                p = p * attention_mask[:, None, None, :]
                entropy = -(p * p.log()).sum(-1)
                entropy = entropy * attention_mask[:, None, :]
                per_example = entropy.sum(-1) / valid_queries[:, None]
                sums[li] += per_example.sum(0).cpu().double()

            count += int(attention_mask.shape[0])

    if sums is None or count == 0:
        raise ValueError("empty calibration loader")

    return [s / count for s in sums]


def von_neumann_entropy(rho: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    eig = torch.linalg.eigvalsh(0.5 * (rho + rho.T)).clamp_min(eps)
    return -(eig * eig.log()).sum()


def select_topk(scores, k):
    return {
        layer: torch.topk(score, k=k, largest=True).indices.tolist()
        for layer, score in enumerate(scores)
    }


def random_select(layers: int, heads: int, k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return {
        layer: torch.randperm(heads, generator=g)[:k].tolist()
        for layer in range(layers)
    }


def mask_from_selection(selected, layers, heads, device):
    mask = torch.zeros((layers, heads), dtype=torch.float32, device=device)
    for layer, selected_heads in selected.items():
        mask[layer, selected_heads] = 1.0
    return mask


def bootstrap_result(
    model,
    loader,
    device,
    baseline_losses,
    baseline_correct,
    head_mask,
    seed,
):
    losses, correct = evaluate_per_example(model, loader, device, head_mask)
    loss_ci = paired_bootstrap_delta(
        losses,
        baseline_losses,
        n_boot=2000,
        seed=seed,
    )
    acc_ci = paired_bootstrap_delta(
        correct,
        baseline_correct,
        n_boot=2000,
        seed=seed + 17,
    )
    return {
        "accuracy": float(correct.mean()),
        "loss": float(losses.mean()),
        "accuracy_delta_vs_full": acc_ci,
        "loss_delta_vs_full": loss_ci,
    }, losses, correct


def main():
    parser = argparse.ArgumentParser(
        description="Full SST-2 evaluation across head budgets."
    )
    parser.add_argument(
        "--model-id",
        default="textattack/bert-base-uncased-SST-2",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=256,
        help="Calibration examples sampled from SST-2 train.",
    )
    parser.add_argument(
        "--heads-to-keep",
        default="9,8,6",
        help="Comma-separated heads per layer (e.g. 9,8,6).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", default="results/step5_full_sst2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = [int(x) for x in args.heads_to_keep.split(",") if x.strip()]
    if not budgets:
        raise ValueError("at least one head budget is required")

    torch.manual_seed(args.seed)

    train = load_dataset("stanfordnlp/sst2", split="train")
    validation = load_dataset("stanfordnlp/sst2", split="validation")
    if args.calibration_size > len(train):
        raise ValueError("calibration-size exceeds SST-2 train split")

    # Calibration is selected only from train. The full validation split is
    # reserved for final evaluation to avoid selection/evaluation overlap.
    calibration = train.shuffle(seed=args.seed).select(range(args.calibration_size))

    model, tokenizer = load_sequence_classifier(args.model_id, args.device)

    calibration_loader = make_text_loader(
        calibration,
        tokenizer,
        text_fields=("sentence",),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    validation_loader = make_text_loader(
        validation,
        tokenizer,
        text_fields=("sentence",),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    baseline_losses, baseline_correct = evaluate_per_example(
        model,
        validation_loader,
        args.device,
    )
    baseline_accuracy = float(baseline_correct.mean())
    baseline_loss = float(baseline_losses.mean())
    baseline_params = parameter_count(model)

    print(
        f"Full validation: accuracy={baseline_accuracy:.6f}, "
        f"loss={baseline_loss:.6f}, examples={len(validation)}"
    )

    print("Collecting QFC calibration states...")
    mean_states = collect_mean_density_states(
        model,
        calibration_loader,
        args.device,
    )
    layers = len(mean_states)
    heads = int(mean_states[0].shape[0])

    print("Computing Shannon/Von-Neumann calibration scores...")
    shannon = shannon_entropy_scores(model, calibration_loader, args.device)
    vn = [
        torch.stack([von_neumann_entropy(rho) for rho in states])
        for states in mean_states
    ]

    print("Computing corrected Michel gate sensitivity...")
    michel = michel_head_importance(
        model,
        calibration_loader,
        args.device,
    ).detach().cpu()
    michel_scores = [michel[layer] for layer in range(layers)]

    all_results = {
        "metadata": {
            "model_id": args.model_id,
            "dataset": "stanfordnlp/sst2",
            "calibration_split": "train",
            "calibration_examples": args.calibration_size,
            "evaluation_split": "validation",
            "evaluation_examples": len(validation),
            "max_length": args.max_length,
            "seed": args.seed,
            "device": args.device,
            "heads_per_layer": heads,
            "layers": layers,
            "baseline": {
                "accuracy": baseline_accuracy,
                "loss": baseline_loss,
                "parameters": baseline_params,
            },
        },
        "budgets": {},
    }

    for k in budgets:
        if not 1 <= k <= heads:
            raise ValueError(f"budget {k} must be between 1 and {heads}")

        qfc_selected = {}
        qfc_cov = {}
        for layer, states in enumerate(mean_states):
            sim = pairwise_fidelity(states)
            selected, history = greedy_select(sim, k)
            qfc_selected[layer] = selected
            qfc_cov[layer] = float(history[-1])

        selections = {
            "QFC": qfc_selected,
            "Shannon": select_topk(shannon, k),
            "VonNeumann": select_topk(vn, k),
            "MichelGate": select_topk(michel_scores, k),
            "Random": random_select(
                layers,
                heads,
                k,
                seed=args.seed + 100 + k,
            ),
        }

        budget_result = {
            "heads_kept_per_layer": k,
            "methods": {},
            "qfc_coverage_total": sum(qfc_cov.values()),
        }

        for method, selected in selections.items():
            mask = mask_from_selection(selected, layers, heads, args.device)
            result, _, _ = bootstrap_result(
                model,
                validation_loader,
                args.device,
                baseline_losses,
                baseline_correct,
                mask,
                seed=args.seed + k * 1000 + len(method),
            )
            result["selected_heads_zero_based"] = selected
            budget_result["methods"][method] = result

        # Physical pruning check only for QFC; head masks above already validate
        # function, while this verifies that the actual surgically reduced model
        # matches the masked implementation.
        physical = copy.deepcopy(model)
        physical.to(args.device)
        physical, pruned = structured_prune(physical, qfc_selected)
        physical_losses, physical_correct = evaluate_per_example(
            physical,
            validation_loader,
            args.device,
        )
        physical_loss_ci = paired_bootstrap_delta(
            physical_losses,
            baseline_losses,
            n_boot=args.bootstrap,
            seed=args.seed + 8000 + k,
        )
        physical_acc_ci = paired_bootstrap_delta(
            physical_correct,
            baseline_correct,
            n_boot=args.bootstrap,
            seed=args.seed + 9000 + k,
        )
        budget_result["QFC_structured"] = {
            "accuracy": float(physical_correct.mean()),
            "loss": float(physical_losses.mean()),
            "accuracy_delta_vs_full": physical_acc_ci,
            "loss_delta_vs_full": physical_loss_ci,
            "parameters": parameter_count(physical),
            "parameter_fraction_remaining": parameter_count(physical) / baseline_params,
            "pruned_heads_zero_based": pruned,
        }

        print(
            f"k={k}: QFC={budget_result['methods']['QFC']['accuracy']:.6f}, "
            f"VNE={budget_result['methods']['VonNeumann']['accuracy']:.6f}, "
            f"Michel={budget_result['methods']['MichelGate']['accuracy']:.6f}, "
            f"Random={budget_result['methods']['Random']['accuracy']:.6f}"
        )
        all_results["budgets"][str(k)] = budget_result

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps(
        {
            "baseline": all_results["metadata"]["baseline"],
            "budgets": {
                k: {
                    method: {
                        "accuracy": info["accuracy"],
                        "loss": info["loss"],
                    }
                    for method, info in budget["methods"].items()
                }
                for k, budget in all_results["budgets"].items()
            },
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
