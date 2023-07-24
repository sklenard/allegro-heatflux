# Allegro-heatflux

This package implements a custom calculation of partial forces used for the heatflux and stress tensors in lammps using automatic differentiation of the atomic energies (see https://arxiv.org/abs/2307.02327).

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

**NOTE:** The deployed model will be **only** compatible with LAMMPS as it outputs dummy virial/stress tensors. The calculation of virial/stress tensors from partial forces will be probably implemented in the future.

