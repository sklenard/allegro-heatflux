from typing import List, Union, Optional
import warnings

import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

@compile_mode("script")
class PartialForceOutput(GraphModuleMixin, torch.nn.Module):
    r"""PartialForceOutput that only gets invoked for deployed models
    (replaced with standard ForceOutput during training)

    Args:
        func: the energy model
        vectorize: the vectorize option to ``torch.autograd.functional.jacobian``,
            false by default since it doesn't work well.
    """

    def __init__(
        self,
        func: GraphModuleMixin,
    ):
        super().__init__()
        self.func = func

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={AtomicDataDict.PER_ATOM_ENERGY_KEY: Irreps("0e")},
            irreps_out=func.irreps_out,
        )

        self.irreps_out[AtomicDataDict.PARTIAL_FORCE_KEY] = Irreps("1o")
        self.irreps_out[AtomicDataDict.FORCE_KEY] = Irreps("1o")
        self.irreps_out[AtomicDataDict.STRESS_KEY] = "3x1o"
        self.irreps_out[AtomicDataDict.VIRIAL_KEY] = "3x1o"

        self.register_buffer("_empty", torch.Tensor())
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        pos = data[AtomicDataDict.POSITIONS_KEY].requires_grad_()
        device = pos.device

        data = self.func(data)

        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_vec.retain_grad()
        
        toten = data[AtomicDataDict.TOTAL_ENERGY_KEY].squeeze(-1)
        toten.backward()

        partial_forces = edge_vec.grad
        forces = - pos.grad

        data[AtomicDataDict.PARTIAL_FORCE_KEY] = partial_forces
        data[AtomicDataDict.FORCE_KEY] = forces

        # Dummy virial & stress
        data[AtomicDataDict.VIRIAL_KEY] = torch.zeros((3,3), device=device)
        data[AtomicDataDict.STRESS_KEY] = torch.zeros((3,3), device=device)

        return data

