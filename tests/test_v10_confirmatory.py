import torch
from qfc.conditional import conditional_fidelity_kernels
def test_v10_kernel():
    x=torch.eye(4).repeat(3,2,1,1)/4
    assert conditional_fidelity_kernels([x])[0].shape==(3,2,2)
