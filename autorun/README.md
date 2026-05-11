# Autorun Commands

This directory now keeps one script per current dataset:

- `cifar10.sh`
- `femnist.sh`
- `imagenet100.sh`
- `stackoverflow.sh`

Each dataset script contains:

- `FedGMR` commands with the selected patience for that dataset.
- Matching ablation commands using the same patience as the corresponding `FedGMR` run.
- Baseline commands, grouped separately inside the same file.

Additional experiments are collected in:

- `extensions.sh`

This file contains:

- fixed-time restoration commands
- commands that apply GMR to other MHFL methods

Legacy scripts were moved to `autorun/archive/`.
