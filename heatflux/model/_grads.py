from nequip.nn import GraphModuleMixin, GradientOutput
from nequip.data import AtomicDataDict

from heatflux.nn import PartialForceOutput as PartialForceOutputModule


def PartialForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and partial forces to a model that predicts energy.
    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.
    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.PARTIAL_FORCE_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force outputs.")
    return PartialForceOutputModule(func=model)
