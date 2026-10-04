from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Iterable

import torch
from datasets import load_dataset

from qfc.coverage import greedy_select, weighted_greedy_select
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import (
    _move_batch,
    collect_mean_attention_matrices,
    collect_mean_density_states,
    load_sequence_classifier,
    make_text_loader,
    michel_head_importance,
    parameter_count,
    structured_prune,
)
from qfc.metrics import evaluate_per_example, paired_bootstrap_delta
from qfc.similarities import hilbert_schmidt_similarity
from qfc.classical_controls import vectorized_cosine_similarity


def shannon_entropy_scores(model, loader: Iterable[dict[str, torch.Tensor]], device: str):
    model.eval()
    sums = None
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            batch.pop("labels", None)
            attention_mask = batch["attention_mask"].float()

            out = model(**batch, output_attentions=True, return_dict=True)
            if out.attentions is None:
                raise RuntimeError("Model did not return attentions")

            if sums is None:
                sums = [torch.zeros(a.shape[1], dtype=torch.float64) for a in out.attentions]

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


def select_coverage_from_similarity(similarities, k):
    selected = {}
    coverage = {}
    for layer, sim in enumerate(similarities):
        heads, history = greedy_select(sim, k)
        selected[layer] = heads
        coverage[layer] = float(history[-1])
    return selected, coverage


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


def run_masked(
    model,
    loader,
    device,
    baseline_losses,
    baseline_correct,
    selected,
    layers,
    heads,
    bootstrap,
    seed,
):
    mask = mask_from_selection(selected, layers, heads, device)
    losses, correct = evaluate_per_example(model, loader, device, head_mask=mask)
    return {
        "accuracy": float(correct.mean()),
        "loss": float(losses.mean()),
        "accuracy_delta_vs_full": paired_bootstrap_delta(
            correct, baseline_correct, n_boot=bootstrap, seed=seed
        ),
        "loss_delta_vs_full": paired_bootstrap_delta(
            losses, baseline_losses, n_boot=bootstrap, seed=seed + 17
        ),
        "selected_heads_zero_based": selected,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-safe full SST-2 validation for QFC and matched baselines."
    )
    parser.add_argument("--model-id", default="textattack/bert-base-uncased-SST-2")
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=256,
        help="Number of calibration examples sampled from the SST-2 training split.",
    )
    parser.add_argument(
        "--heads-to-keep",
        default="9,8,6",
        help="Comma-separated heads retained per layer.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--random-replicates", type=int, default=3)
    parser.add_argument("--output-dir", default="results/step5_full_sst2")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    budgets = [int(x) for x in args.heads_to_keep.split(",") if x.strip()]
    if not budgets:
        raise ValueError("no pruning budgets supplied")

    torch.manual_seed(args.seed)

    train = load_dataset("stanfordnlp/sst2", split="train")
    validation = load_dataset("stanfordnlp/sst2", split="validation")
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
        model, validation_loader, args.device
    )
    baseline = {
        "accuracy": float(baseline_correct.mean()),
        "loss": float(baseline_losses.mean()),
        "parameters": parameter_count(model),
        "examples": len(validation),
    }
    print(
        f"Full validation: accuracy={baseline['accuracy']:.6f} "
        f"loss={baseline['loss']:.6f} examples={baseline['examples']}"
    )

    print("Collecting calibration density operators...")
    mean_states = collect_mean_density_states(
        model, calibration_loader, args.device
    )
    layers = len(mean_states)
    heads = int(mean_states[0].shape[0])

    print("Collecting mean attention matrices for the classical full-attention control...")
    mean_attention = collect_mean_attention_matrices(model, calibration_loader, args.device)

    print("Computing Shannon/Von-Neumann scores...")
    shannon = shannon_entropy_scores(model, calibration_loader, args.device)
    vn = [
        torch.stack([von_neumann_entropy(rho) for rho in states])
        for states in mean_states
    ]

    print("Computing corrected Michel gate sensitivity...")
    michel = michel_head_importance(
        model, calibration_loader, args.device
    ).detach().cpu()
    michel_scores = [michel[layer] for layer in range(layers)]

    print("Building quantum and classical similarity kernels...")
    fidelity_similarities = [pairwise_fidelity(states) for states in mean_states]
    hs_similarities = [hilbert_schmidt_similarity(states) for states in mean_states]
    cosine_similarities = [vectorized_cosine_similarity(attn) for attn in mean_attention]

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
            "layers": layers,
            "heads_per_layer": heads,
            "baseline": baseline,
            "random_replicates": args.random_replicates,
            "research_note": (
                "This run is the first leakage-safe full-validation experiment. "
                "No superiority claim is made unless confidence intervals and repeated "
                "runs support it."
            ),
        },
        "budgets": {},
    }

    for k in budgets:
        if not 1 <= k <= heads:
            raise ValueError(f"heads-to-keep={k} must be between 1 and {heads}")

        qfc, qfc_cov = select_coverage_from_similarity(
            fidelity_similarities, k
        )
        iwqfc = {}
        iwqfc_cov = {}
        for layer, states in enumerate(mean_states):
            sim = fidelity_similarities[layer]
            raw = michel_scores[layer].clamp_min(0.0)
            if float(raw.sum()) <= 0.0:
                quality = torch.full_like(raw, 1.0 / len(raw))
            else:
                quality = raw / raw.sum()
            selected, history = weighted_greedy_select(sim, quality, k)
            iwqfc[layer] = selected
            iwqfc_cov[layer] = float(history[-1])
        hs_cov_selected, hs_cov = select_coverage_from_similarity(
            hs_similarities, k
        )
        cosine_selected, cosine_cov = select_coverage_from_similarity(
            cosine_similarities, k
        )

        selections = {
            "QFC": qfc,
            "IWQFC": iwqfc,
            "HS_Coverage": hs_cov_selected,
            "Shannon": select_topk(shannon, k),
            "VonNeumann": select_topk(vn, k),
            "MichelGate": select_topk(michel_scores, k),
            "CosineCoverage": cosine_selected,
        }

        budget_result = {
            "heads_kept_per_layer": k,
            "qfc_fidelity_coverage_total": sum(qfc_cov.values()),
            "iwqfc_weighted_coverage_total": sum(iwqfc_cov.values()),
            "hs_coverage_total": sum(hs_cov.values()),
            "cosine_coverage_total": sum(cosine_cov.values()),
            "methods": {},
        }

        for method, selected in selections.items():
            budget_result["methods"][method] = run_masked(
                model,
                validation_loader,
                args.device,
                baseline_losses,
                baseline_correct,
                selected,
                layers,
                heads,
                args.bootstrap,
                seed=args.seed + k * 100 + len(method),
            )

        random_runs = []
        for r in range(args.random_replicates):
            random_selected = random_select(
                layers,
                heads,
                k,
                seed=args.seed + 10000 + 100 * k + r,
            )
            random_runs.append(
                run_masked(
                    model,
                    validation_loader,
                    args.device,
                    baseline_losses,
                    baseline_correct,
                    random_selected,
                    layers,
                    heads,
                    args.bootstrap,
                    seed=args.seed + 20000 + 100 * k + r,
                )
            )

        budget_result["Random"] = {
            "replicates": random_runs,
            "mean_accuracy": sum(x["accuracy"] for x in random_runs) / len(random_runs),
            "mean_loss": sum(x["loss"] for x in random_runs) / len(random_runs),
        }

        # Physical pruning validates the actual architecture surgery for QFC.
        physical = copy.deepcopy(model).to(args.device)
        physical, pruned = structured_prune(physical, iwqfc)
        physical_losses, physical_correct = evaluate_per_example(
            physical, validation_loader, args.device
        )
        budget_result["IWQFC_structured"] = {
            "accuracy": float(physical_correct.mean()),
            "loss": float(physical_losses.mean()),
            "accuracy_delta_vs_full": paired_bootstrap_delta(
                physical_correct,
                baseline_correct,
                n_boot=args.bootstrap,
                seed=args.seed + 30000 + k,
            ),
            "loss_delta_vs_full": paired_bootstrap_delta(
                physical_losses,
                baseline_losses,
                n_boot=args.bootstrap,
                seed=args.seed + 40000 + k,
            ),
            "parameters": parameter_count(physical),
            "parameter_fraction_remaining": parameter_count(physical) / baseline["parameters"],
            "pruned_heads_zero_based": pruned,
        }

        print(
            f"k={k}: "
            f"QFC={budget_result['methods']['QFC']['accuracy']:.6f} "
            f"IWQFC={budget_result['methods']['IWQFC']['accuracy']:.6f} "
            f"HS={budget_result['methods']['HS_Coverage']['accuracy']:.6f} "
            f"Cosine={budget_result['methods']['CosineCoverage']['accuracy']:.6f} "
            f"VNE={budget_result['methods']['VonNeumann']['accuracy']:.6f} "
            f"Michel={budget_result['methods']['MichelGate']['accuracy']:.6f} "
            f"RandomMean={budget_result['Random']['mean_accuracy']:.6f}"
        )

        all_results["budgets"][str(k)] = budget_result

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps(all_results["metadata"]["baseline"], indent=2))


if __name__ == "__main__":
    main()
