from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset

from qfc.coverage import greedy_select
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import (
    _move_batch,
    collect_mean_density_states,
    load_sequence_classifier,
    make_text_loader,
)


@torch.no_grad()
def eval_metrics(model, loader, device, head_mask=None):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        out = model(
            **batch,
            labels=labels,
            head_mask=head_mask,
            return_dict=True,
        )
        n = int(labels.numel())
        total += n
        correct += int((out.logits.argmax(-1) == labels).sum().item())
        loss_sum += float(out.loss.item()) * n
    return {
        "accuracy": correct / total,
        "loss": loss_sum / total,
    }


@torch.no_grad()
def mean_shannon(model, loader, device):
    model.eval()
    sums = None
    count = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        batch.pop("labels", None)
        mask = batch["attention_mask"].float()
        out = model(**batch, output_attentions=True, return_dict=True)
        if out.attentions is None:
            raise RuntimeError("Model did not return attentions")

        if sums is None:
            sums = [torch.zeros(a.shape[1], dtype=torch.float64) for a in out.attentions]

        valid_queries = mask.sum(dim=-1).clamp_min(1.0)
        for li, attn in enumerate(out.attentions):
            p = attn.float().clamp_min(1e-12)
            p = p * mask[:, None, None, :]
            h = -(p * p.clamp_min(1e-12).log()).sum(-1)
            h = h * mask[:, None, :]
            per_example = h.sum(-1) / valid_queries[:, None]
            sums[li] += per_example.sum(0).cpu().double()
        count += int(mask.shape[0])

    if sums is None or count == 0:
        raise ValueError("empty calibration loader")
    return [s / count for s in sums]


def vn_entropy(rho, eps=1e-12):
    eig = torch.linalg.eigvalsh(0.5 * (rho + rho.T)).clamp_min(eps)
    return -(eig * eig.log()).sum()


def topk(scores, k):
    return {i: torch.topk(s, k).indices.tolist() for i, s in enumerate(scores)}


def random_selection(layers, heads, k, seed):
    g = torch.Generator().manual_seed(seed)
    return {i: torch.randperm(heads, generator=g)[:k].tolist() for i in range(layers)}


def qfc_selection(mean_states, k):
    selected = {}
    coverage = {}
    for li, states in enumerate(mean_states):
        sim = pairwise_fidelity(states)
        heads, history = greedy_select(sim, k)
        selected[li] = heads
        coverage[li] = float(history[-1])
    return selected, coverage


def mask_from(selected, layers, heads, device):
    mask = torch.zeros((layers, heads), dtype=torch.float32, device=device)
    for li, hs in selected.items():
        mask[li, hs] = 1.0
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="textattack/bert-base-uncased-SST-2")
    p.add_argument("--offsets", default="0,96,192")
    p.add_argument("--calibration-size", type=int, default=32)
    p.add_argument("--evaluation-size", type=int, default=32)
    p.add_argument("--heads-to-keep", type=int, default=9)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--output-dir", default="results/step4_stability")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dataset = load_dataset("stanfordnlp/sst2", split="validation")
    offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
    model, tokenizer = load_sequence_classifier(args.model_id, args.device)

    all_results = []
    for split_id, offset in enumerate(offsets):
        end_cal = offset + args.calibration_size
        end_eval = end_cal + args.evaluation_size
        if end_eval > len(dataset):
            raise ValueError(f"split at offset {offset} exceeds validation size {len(dataset)}")

        cal = dataset.select(range(offset, end_cal))
        eva = dataset.select(range(end_cal, end_eval))

        cal_loader = make_text_loader(
            cal, tokenizer, ("sentence",),
            batch_size=args.batch_size, max_length=args.max_length
        )
        eval_loader = make_text_loader(
            eva, tokenizer, ("sentence",),
            batch_size=args.batch_size, max_length=args.max_length
        )

        base = eval_metrics(model, eval_loader, args.device)
        states = collect_mean_density_states(model, cal_loader, args.device)
        sh = mean_shannon(model, cal_loader, args.device)
        vn = [torch.stack([vn_entropy(r) for r in states]) for states in states]
        layers = len(states)
        heads = int(states[0].shape[0])

        qfc, qfc_cov = qfc_selection(states, args.heads_to_keep)
        sh_sel = topk(sh, args.heads_to_keep)
        vn_sel = topk(vn, args.heads_to_keep)
        rnd_sel = random_selection(layers, heads, args.heads_to_keep, seed=1000 + split_id)

        selections = {
            "QFC": qfc,
            "Shannon": sh_sel,
            "VonNeumann": vn_sel,
            "Random": rnd_sel,
        }

        split_results = {
            "split_id": split_id,
            "offset": offset,
            "calibration_size": args.calibration_size,
            "evaluation_size": args.evaluation_size,
            "baseline": base,
            "methods": {},
            "qfc_coverage_sum": sum(qfc_cov.values()),
        }

        for name, selection in selections.items():
            m = mask_from(selection, layers, heads, args.device)
            metric = eval_metrics(model, eval_loader, args.device, head_mask=m)
            metric["accuracy_change_vs_full"] = metric["accuracy"] - base["accuracy"]
            metric["loss_change_vs_full"] = metric["loss"] - base["loss"]
            metric["selected_heads_zero_based"] = selection
            split_results["methods"][name] = metric

        # Pairwise overlap of QFC and Von Neumann selections.
        jaccards = []
        for li in range(layers):
            a = set(qfc[li])
            b = set(vn_sel[li])
            jaccards.append(len(a & b) / len(a | b))
        split_results["qfc_vn_jaccard_mean"] = sum(jaccards) / len(jaccards)

        print(
            f"split={split_id} offset={offset} "
            f"full_acc={base['accuracy']:.4f} "
            f"qfc_acc={split_results['methods']['QFC']['accuracy']:.4f} "
            f"vn_acc={split_results['methods']['VonNeumann']['accuracy']:.4f}"
        )
        all_results.append(split_results)

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    payload = {
        "model_id": args.model_id,
        "device": args.device,
        "heads_to_keep_per_layer": args.heads_to_keep,
        "splits": all_results,
        "research_note": (
            "Small-split engineering stability study. It is not a paper result until "
            "the full validation split and contemporary baselines are evaluated."
        ),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Compact aggregate for quick inspection.
    for method in ["QFC", "Shannon", "VonNeumann", "Random"]:
        accs = [s["methods"][method]["accuracy"] for s in all_results]
        losses = [s["methods"][method]["loss"] for s in all_results]
        print(
            method,
            "mean_acc=", sum(accs) / len(accs),
            "mean_loss=", sum(losses) / len(losses),
        )


if __name__ == "__main__":
    main()
