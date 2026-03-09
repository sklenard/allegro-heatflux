# Allegro-heatflux

This package implements a custom calculation of edge forces used for the heatflux and stress tensors in lammps using automatic differentiation of the atomic energies (see https://link.aps.org/doi/10.1103/PhysRevMaterials.8.033802).

**NOTE:** This package is compatible with [Allegro](https://github.com/mir-group/allegro) commit [`fa538bc`](https://github.com/mir-group/allegro/commit/fa538bc) (NequIP 0.6.1). It should **not** be used with NequIP-based models with more than 1 message-passing step, as contributions from atoms beyond the cutoff would be missed.

## Installation

Clone the git repository and add the folder in your `$PYTHONPATH` environment variable.

## Usage
The Allegro model has to be trained normally including the `StressForceOutput` in the `model_builders` option (See the [Allegro README](https://github.com/mir-group/allegro#usage)):

For example:
```yaml
model_builders:
 - allegro.model.Allegro
 - PerSpeciesRescale
 - StressForceOutput
 - RescaleEnergyEtc
```

For the deploy, `StressForceOutput` in the `config.yaml` file (from the `root/run_name` folder) has to be replaced by `heatflux.model.PartialForceOutput`. This can be done automatically using the following commands (after setting the `${train_folder}` variable):

```bash
cp ${train_folder}/config.yaml ${train_folder}/config.yaml.bak
sed -i 's/StressForceOutput/heatflux.model.PartialForceOutput/g' ${train_folder}/config.yaml
nequip-deploy build --train-dir ${train_folder} model-lmp.pth
cp ${train_folder}/config.yaml.bak ${train_folder}/config.yaml
```

**NOTE:** The deployed model will be **only** compatible with LAMMPS as it outputs a zero (dummy) virial/stress tensor — the actual stress is computed from the edge forces directly in LAMMPS.

