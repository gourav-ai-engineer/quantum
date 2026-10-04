import torch

from qfc.conditional import conditional_fidelity_kernels


def test_conditional_fidelity_pipeline_shapes():
    states = torch.eye(4).repeat(5, 3, 1, 1).float()
    states = states / states.shape[-1]
    kernels = conditional_fidelity_kernels([states])
    assert len(kernels) == 1
    assert kernels[0].shape == (5, 3, 3)
    assert torch.allclose(
        torch.diagonal(kernels[0], dim1=-2, dim2=-1),
        torch.ones(5, 3),
        atol=1e-5,
    )
