from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from .coverage import greedy_select
from .fidelity import pairwise_fidelity
from .states import aggregate_density_operators, density_operator


def _require_transformers():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional transformer stack with: "
            "pip install -e '.[transformers]'"
        ) from exc
    return AutoModelForSequenceClassification, AutoTokenizer


def load_sequence_classifier(model_id: str, device: str = "cpu"):
    """Load a sequence classifier with eager attention when supported."""
    AutoModelForSequenceClassification, AutoTokenizer = _require_transformers()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, attn_implementation="eager"
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

    model.to(device)
    return model, tokenizer


def make_text_loader(
    dataset,
    tokenizer,
    text_fields: tuple[str, ...],
    label_field: str = "label",
    batch_size: int = 16,
    max_length: int = 128,
    max_samples: int | None = None,
) -> DataLoader:
    """Create a fixed-length tokenizer DataLoader.

    Fixed padding is deliberate: density operators for different examples must
    live in the same matrix dimension for aggregation and pairwise comparison.
    """
    rows = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    def collate(examples):
        texts = []
        for row in examples:
            if len(text_fields) == 1:
                texts.append(row[text_fields[0]])
            else:
                texts.append(tuple(row[name] for name in text_fields))

        if len(text_fields) == 1:
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            first = [x[0] for x in texts]
            second = [x[1] for x in texts]
            encoded = tokenizer(
                first,
                second,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        encoded["labels"] = torch.tensor(
            [int(row[label_field]) for row in examples], dtype=torch.long
        )
        return encoded

    return DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collate)


def _move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_accuracy(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    device: str = "cpu",
    head_mask: torch.Tensor | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        outputs = model(
            **batch,
            labels=labels,
            head_mask=head_mask,
            return_dict=True,
        )
        correct += int((outputs.logits.argmax(dim=-1) == labels).sum().item())
        total += int(labels.numel())

    if total == 0:
        raise ValueError("evaluation loader contains no examples")
    return correct / total


@torch.no_grad()
def collect_mean_density_states(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Collect mean density operators for every layer and head.

    Returns a list of L tensors, each shaped [H, T, T].
    """
    model.eval()
    sums = None
    total = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        batch_attention_mask = batch["attention_mask"].to(torch.float32)
        batch.pop("labels", None)

        outputs = model(
            **batch,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("Model did not return attention tensors")

        if sums is None:
            sums = [
                torch.zeros(
                    (attn.shape[1], attn.shape[-1], attn.shape[-1]),
                    dtype=torch.float64,
                    device="cpu",
                )
                for attn in attentions
            ]

        batch_size = attentions[0].shape[0]
        for layer_index, attn in enumerate(attentions):
            # Remove padded token rows/columns before forming the Gram matrix.
            token_mask = batch_attention_mask.to(attn.dtype)
            masked = attn * token_mask[:, None, :, None] * token_mask[:, None, None, :]
            rho = density_operator(masked)
            sums[layer_index] += rho.sum(dim=0).cpu().double()

        total += batch_size

    if total == 0 or sums is None:
        raise ValueError("calibration loader contains no examples")

    return [layer_sum / total for layer_sum in sums]


def qfc_select_from_mean_states(
    mean_states: list[torch.Tensor],
    heads_to_keep: int,
) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """Select heads independently within each layer using fidelity coverage."""
    selected_by_layer: dict[int, list[int]] = {}
    history_by_layer: dict[int, list[float]] = {}

    for layer_index, states in enumerate(mean_states):
        sim = pairwise_fidelity(states)
        selected, history = greedy_select(sim, heads_to_keep)
        selected_by_layer[layer_index] = selected
        history_by_layer[layer_index] = [float(x) for x in history]

    return selected_by_layer, history_by_layer


def save_selection(
    output_dir: str | Path,
    selected_by_layer: dict[int, list[int]],
    history_by_layer: dict[int, list[float]],
    mean_states: list[torch.Tensor],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "selection.json").write_text(
        json.dumps(
            {
                "selected_heads_zero_based": selected_by_layer,
                "coverage_history": history_by_layer,
            },
            indent=2,
        )
    )
    torch.save([state.float() for state in mean_states], out / "mean_density_states.pt")


def michel_head_importance(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    device: str = "cpu",
    max_batches: int | None = None,
) -> torch.Tensor:
    """Michel-style gate sensitivity: E_x |dL/dg_h|.

    The head mask is kept at one and treated as a differentiable gate. This is
    intentionally separate from QFC so the baseline has an explicit loss
    sensitivity interpretation.
    """
    model.eval()
    base_model = getattr(model, model.base_model_prefix, model)

    num_layers = int(base_model.config.num_hidden_layers)
    num_heads = int(base_model.config.num_attention_heads)
    head_mask = torch.ones(
        (num_layers, num_heads), dtype=torch.float32, device=device, requires_grad=True
    )

    total_weighted = torch.zeros_like(head_mask)
    total_examples = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        model.zero_grad(set_to_none=True)
        if head_mask.grad is not None:
            head_mask.grad.zero_()

        outputs = model(
            **batch,
            labels=labels,
            head_mask=head_mask,
            return_dict=True,
        )
        outputs.loss.backward()

        if head_mask.grad is None:
            raise RuntimeError(
                "No gradient reached head_mask. Check the Transformers version/model backend."
            )

        batch_size = int(labels.shape[0])
        total_weighted += head_mask.grad.detach().abs() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("importance loader contains no examples")

    return total_weighted / total_examples


def structured_prune(model, selected_by_layer: dict[int, list[int]]):
    """Physically remove all heads not selected in each layer."""
    base_model = getattr(model, model.base_model_prefix, model)
    num_heads = int(base_model.config.num_attention_heads)

    prune_dict: dict[int, list[int]] = {}
    for layer, selected in selected_by_layer.items():
        selected_set = set(selected)
        prune_dict[layer] = [h for h in range(num_heads) if h not in selected_set]

    if hasattr(base_model, "prune_heads"):
        base_model.prune_heads(prune_dict)
    elif hasattr(model, "prune_heads"):
        model.prune_heads(prune_dict)
    else:
        raise RuntimeError("This model does not expose a structured prune_heads API")

    return model, prune_dict


def parameter_count(model) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def head_mask_from_selection(
    num_layers: int,
    num_heads: int,
    selected_by_layer: dict[int, list[int]],
    device: str,
) -> torch.Tensor:
    mask = torch.zeros((num_layers, num_heads), dtype=torch.float32, device=device)
    for layer, selected in selected_by_layer.items():
        mask[layer, selected] = 1.0
    return mask


@torch.no_grad()
def collect_mean_attention_matrices(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Collect mean masked attention matrices for each layer/head.

    Returns a list of L tensors shaped [H, T, T].
    """
    model.eval()
    sums = None
    total = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch.pop("labels", None)
        attention_mask = batch["attention_mask"].to(torch.float32)

        outputs = model(
            **batch,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("Model did not return attention tensors")

        if sums is None:
            sums = [
                torch.zeros(
                    (attn.shape[1], attn.shape[-1], attn.shape[-1]),
                    dtype=torch.float64,
                )
                for attn in attentions
            ]

        token_mask = attention_mask.to(attentions[0].dtype)
        for layer_idx, attn in enumerate(attentions):
            masked = attn * token_mask[:, None, :, None] * token_mask[:, None, None, :]
            sums[layer_idx] += masked.sum(dim=0).cpu().double()

        total += int(attentions[0].shape[0])

    if sums is None or total == 0:
        raise ValueError("empty calibration loader")

    return [layer_sum / total for layer_sum in sums]
