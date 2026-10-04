import torch

from qfc.classical_controls import vectorized_cosine_similarity
from qfc.fidelity import pairwise_fidelity
from qfc.similarities import hilbert_schmidt_similarity


def test_representation_kernels_have_unit_diagonal_and_valid_range():
    states = torch.eye(3).repeat(4, 1, 1).double()
    states = states / states.shape[-1]
    attn = torch.rand(4, 3, 3).double()
    attn = attn / attn.sum(-1, keepdim=True)

    for sim in (
        pairwise_fidelity(states),
        hilbert_schmidt_similarity(states),
        vectorized_cosine_similarity(attn),
    ):
        assert sim.shape == (4, 4)
        assert torch.all(sim >= 0)
        assert torch.all(sim <= 1)
        assert torch.allclose(
            torch.diag(sim),
            torch.ones(4, dtype=sim.dtype),
    )


def test_hilbert_schmidt_matches_identity_geometry():
    states = torch.eye(3).repeat(2, 1, 1).double() / 3
    sim = hilbert_schmidt_similarity(states)
    assert torch.allclose(
        sim,
        torch.ones(2, 2, dtype=sim.dtype),
    )