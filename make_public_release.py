#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
from pathlib import Path


EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "results",
    "rebuttal_github_bundle",
}

EXCLUDE_FILE_PATTERNS = [
    "REBUTTAL*.md",
    "AC_CONFIDENTIAL*.md",
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.fdb_latexmk",
    "*.fls",
    "*.log",
    "*.out",
    "*.synctex.gz",
]

# Paper source and submission-only assets should not be public.
EXCLUDE_PATH_PATTERNS = [
    "FedGMR/*.tex",
    "FedGMR/*.bib",
    "FedGMR/*.cls",
    "FedGMR/*.sty",
    "FedGMR/*.bst",
    "FedGMR/example_paper.*",
    "FedGMR/appendix*",
]

# Code- and figure-related top-level items we keep by default.
TOP_LEVEL_INCLUDE = {
    "bases",
    "configs",
    "experiments",
    "figure",
    "FedGMR",
    "datasets",
    "autorun",
    "control",
    "utils",
    "Feminst_final.ipynb",
    "auto_run2.py",
    "make_public_release.py",
    "README.md",
    "requirements.txt",
    "environment.yml",
    "setup.py",
    "setenv.sh",
}


PUBLIC_README = """# FedGMR Public Release

Accepted at ICML 2026.

FedGMR studies a stage-dependent problem in model-heterogeneous federated learning (MHFL): small sub-models help bandwidth-constrained clients participate efficiently in the early stage, but their contribution fades later as model capacity becomes the bottleneck. FedGMR addresses this issue through **Gradual Model Restoration (GMR)**, which progressively restores client model density during training so that bandwidth-constrained clients remain useful contributors in the late stage as well.

## Project Overview

In heterogeneous federated learning deployments, bandwidth-constrained clients often struggle to contribute effectively throughout training. Existing MHFL methods usually assign fixed small sub-models to weak clients to reduce communication and training cost. This works well early on, but fixed low-density sub-models become increasingly under-parameterized later in training, causing their updates to become weak, noisy, or uninformative.

FedGMR is built around a simple observation: **small sub-models are useful early, but should not stay fixed forever**. The framework gradually restores client model density during training, so the same resource-constrained clients can train quickly at first and still contribute meaningful updates later.

Around this core mechanism, the project also includes:

- asynchronous coordination for heterogeneous clients,
- mask-aware aggregation to stabilize restoration,
- empirical analysis of stage-dependent model utility,
- convergence analysis for MHFL under incomplete sub-model updates.

## Main Contributions

- **Stage-dependent MHFL insight.** We identify that fixed low-density sub-models are effective in the early stage but lose effectiveness later, even if they still allow frequent participation.
- **Gradual Model Restoration.** FedGMR progressively restores model density during training, re-activating bandwidth-constrained clients in the late stage.
- **Stability-aware aggregation.** Mask-aware aggregation is designed to better preserve gradient stability during restoration, making it more compatible with GMR than naive aggregation.
- **Theory for heterogeneous sub-model training.** The analysis characterizes the bias and variance introduced by incomplete client updates, highlights the role of average density and coverage, and shows that GMR narrows the optimization gap toward full-model FL.
- **Cross-method generality.** GMR is effective not only in the proposed framework but also when applied on top of other MHFL methods.

## Main Figures

### 1. FedGMR framework and core idea

This is the main idea figure of the paper. FedGMR starts from the observation that bandwidth-constrained clients can benefit from small sub-models early, but should gradually recover model capacity later to remain useful global contributors.

![FedGMR framework](FedGMR/GMR_framework.png)

### 2. Stage-dependent benefit of model density

This figure illustrates the central empirical observation behind FedGMR: smaller sub-models improve faster early, while larger-capacity models become more beneficial later.

![Learning-rate analysis](FedGMR/learning_rate_different_density_models_strip.png)

### 3. Cross-method gains from GMR

Applying GMR to multiple MHFL baselines consistently improves performance. This supports that the main contribution is the **restoration mechanism itself**, rather than one specific pruning rule or one specific base method.

![Cross-method comparison](FedGMR/GMR_on_other_MHFL_methods.png)

### 4. Aggregation and ablation analysis

The ablation results highlight that restoration and aggregation are coupled: GMR improves late-stage usefulness, while mask-aware aggregation helps maintain stability during restoration.

![Ablation analysis](FedGMR/Ablation_method.png)

### 5. Robustness to restoration timing

FedGMR does not depend on one exact restoration trigger. A fixed-time restoration variant remains effective across datasets, although adaptive triggering can further improve some settings.

![Fixed-time restoration](FedGMR/Fixed_restoration_timeing_plot.png)

## Representative Results

### Main baseline comparison under high heterogeneity

The table below summarizes the main baseline comparison from the paper under the hardest setting, where the gain from gradual restoration is most visible.

| Method | FEMNIST IID | FEMNIST Non-IID | CIFAR-10 IID | CIFAR-10 Non-IID | ImageNet-100 IID | ImageNet-100 Non-IID | StackOverflow IID | StackOverflow Non-IID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FedGMR** | **82.67** | **81.86** | **84.52** | **81.68** | **60.84** | **58.01** | **30.00** | **30.07** |
| FedAvg | 75.62 | 74.64 | 70.75 | 68.46 | 48.65 | 47.46 | 24.38 | 24.38 |
| FedAsync | 81.34 | 81.03 | 82.86 | 79.28 | 58.19 | 55.36 | 25.86 | 25.83 |
| HeteroFL | 82.22 | 79.80 | 81.06 | 75.64 | 41.90 | 28.01 | 29.42 | 29.17 |
| FedRolex | 82.19 | 77.83 | 80.85 | 67.82 | 32.17 | 15.26 | 25.39 | 25.15 |
| FjORD | 82.59 | 81.85 | 81.79 | 81.34 | 41.18 | 32.93 | 29.18 | 27.73 |
| FIARSE | 81.22 | 78.77 | 74.04 | 69.19 | 54.00 | 48.27 | 28.77 | 28.57 |

Under high heterogeneity, FedGMR achieves the strongest gains on the hardest settings, especially on CIFAR-10 and ImageNet-100 under Non-IID splits. This is where fixed low-density sub-models suffer the most from late-stage capacity limits.

### Cross-method gains on FEMNIST

These representative results show that adding GMR on top of different MHFL methods yields consistent improvements.

| Method | Base | +Aux | +Aux+GMR |
|---|---:|---:|---:|
| HeteroFL | 79.80 | 80.42 | 82.29 |
| FjORD | 81.76 | 82.16 | 82.20 |
| FedRolex | 77.83 | 80.36 | 81.58 |
| FIARSE | 78.77 | 79.56 | 81.97 |

### Fixed-time restoration still works

The fixed-time variant removes server-side early-stopping/stagnation triggering and restores directly according to training progress. It still improves over `w/o GMR`, which supports that the value of GMR does not depend on one exact trigger.

| Dataset | GMR (ES) | GMR (Fixed) | w/o Asyn | w/o GMR |
|---|---:|---:|---:|---:|
| FEMNIST | 81.71 ± 0.20 | 80.07 ± 0.32 | 78.80 ± 0.54 | 79.51 ± 0.35 |
| CIFAR-10 | 81.68 ± 0.28 | 80.42 ± 0.48 | 80.53 ± 0.20 | 71.93 ± 0.45 |
| ImageNet100 | 58.01 ± 0.50 | 56.38 ± 0.93 | 57.85 ± 1.50 | 48.28 ± 0.73 |
| StackOverflow | 30.04 ± 0.008 | 30.15 ± 0.018 | 30.11 ± 0.034 | 29.76 ± 0.019 |

## Repository Contents

This public release includes:

- core training and aggregation code,
- experiment scripts and configuration files,
- selected visualization assets under `figure/` and `FedGMR/`,
- root-level notebooks used for plotting and result analysis,
- `control/` and `utils/` modules required by experiment entrypoints,
- `setenv.sh` for environment setup.

This public release excludes:

- paper source files (`.tex`, `.bib`, style files, submission-only assets),
- rebuttal documents and confidential review comments,
- large non-public result directories.

## How to Run

Before running any experiment, source the environment helper:

```bash
source setenv.sh
```

The public release expects `datasets/` to point to prepared data storage. In this release it is typically used as a lightweight placeholder or symlink target rather than a fully bundled data directory.

## Dataset Preparation

The first run of a dataset may trigger an automatic download or preprocessing step. If automatic download is not supported, place the raw files in the expected directory and rerun the same command; the code will then convert the raw data into the required format.

- `CIFAR10`: downloaded automatically through `torchvision`.
- `FEMNIST`: when `download=True`, the project clones LEAF and preprocesses FEMNIST automatically. If the raw data already exists, rerunning will process it into the required format.
- `StackOverflow`: if TensorFlow Federated is installed, the project can build `stackoverflow_train.pt`, `stackoverflow_val.pt`, and `stackoverflow_vocab.pt` automatically on first use.
- `ImageNet100`: automatic download is not provided. After you manually place the raw `ILSVRC` files under `datasets/ImageNet100/ILSVRC`, rerunning the same command will automatically convert them into LMDB.

## Example Commands

The commands below are representative examples extracted from the existing `autorun/*.sh` scripts and the notebook-based comparison sweeps.

### CIFAR10

```bash
source setenv.sh
python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid
python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client medium -patience 5 -niid
```

Ablation-style examples from the shell scripts use the same patience values as the paired comparison settings:

```bash
python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client medium -patience 5 -niid
python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex asyn -num_clients 10 -sample_client medium -patience 25 -bp
python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 25 -bp
python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex gradient_avg -num_clients 10 -sample_client medium -patience 25 -bp
python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 25 -bp -re
```

The notebook-based CIFAR10 comparison sweep uses the following patience settings:

- IID:
  - high: `pr_fl` `10`, `fed_avg` `5`, `fed_asyn` `5`, `heterofl` `5`, `fedrolex` `5`, `fjord` `5`, `fiarse` `10`
  - medium: `pr_fl` `25`, `fed_avg` `5`, `fed_asyn` `5`, `heterofl` `5`, `fedrolex` `5`, `fjord` `5`, `fiarse` `10`
  - low: `pr_fl` `1`, `fed_asyn` `5`, `heterofl` `5`, `fedrolex` `5`, `fjord` `5`, `fiarse` `10`
- Non-IID:
  - high: `pr_fl` `15`, `fed_avg` `5`, `fed_asyn` `5`, `heterofl` `5`, `fedrolex` `5`, `fjord` `5`, `fiarse` `10`
  - medium: `pr_fl` `30`, `fed_avg` `5`, `fed_asyn` `5`, `heterofl` `5`, `fedrolex` `5`, `fjord` `5`, `fiarse` `10`
  - low: `pr_fl` `1`, `fed_avg` `5`, `fed_asyn` `5`, `heterofl` `5`, `fedrolex` `5`, `fjord` `5`, `fiarse` `10`

### FEMNIST

```bash
source setenv.sh
python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -num_clients 10 -sample_client high -patience 5 -niid -bp --recover_step_mode fixed --recover_step 0.1
python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex gmr_fiarse -num_clients 10 -sample_client high -patience 10 -niid -bp --recover_step_mode ladder
```

Ablation-style FEMNIST examples from the shell scripts also keep the same patience as the paired runs:

```bash
python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -num_clients 10 -sample_client high -patience 5 -niid -bp
python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client high -patience 7 -niid -bp
python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg -num_clients 10 -sample_client high -patience 7 -niid -bp
python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg -num_clients 10 -sample_client high -patience 7 -niid -bp
python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high -patience 7 -niid -bp -re
```

The notebook-based FEMNIST comparison sweep uses the following patience settings:

- IID:
  - high: `pr_fl=5`, `fed_avg=5`, `fed_asyn=5`, `heterofl=5`, `fedrolex=10`, `fjord=5`, `fiarse=5`
  - medium: `pr_fl=5`, `fed_avg=5`, `fed_asyn=5`, `heterofl=5`, `fedrolex=10`, `fjord=5`, `fiarse=5`
  - low: `pr_fl=3`, `fed_avg=5`, `fed_asyn=5`, `heterofl=5`, `fedrolex=10`, `fjord=5`, `fiarse=5`
- Non-IID:
  - high: `pr_fl=7`, `fed_avg=5`, `fed_asyn=5`, `heterofl=5`, `fedrolex=10`, `fjord=5`, `fiarse=5`
  - medium: `pr_fl=3`, `fed_avg=5`, `fed_asyn=5`, `heterofl=5`, `fedrolex=10`, `fjord=5`, `fiarse=5`
  - low: `pr_fl=1`, `fed_avg=5`, `fed_asyn=5`, `heterofl=5`, `fedrolex=10`, `fjord=5`, `fiarse=5`

### ImageNet100

```bash
source setenv.sh
python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid
python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10
```

Ablation-style ImageNet100 examples from the shell scripts keep the same patience as the paired runs:

```bash
python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -ac wg -num_clients 10 -sample_client medium -patience 5 -niid
python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client medium -patience 5 -niid
python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid
python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client medium -patience 10 -niid
```

The notebook-based ImageNet100 comparison sweep uses the following `pr_fl` patience settings:

- IID:
  - high: `40`
  - medium: `30`
  - low: `40`
- Non-IID:
  - high: `25`
  - medium: `30`
  - low: `30`

The remaining baselines in the notebook use the exact experiment names shown in the notebook cells, while `pr_fl` is the one that explicitly encodes patience.

### StackOverflow

```bash
source setenv.sh
python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 70000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0
python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -num_clients 10 -sample_client high -patience 10 -niid
```

Ablation-style StackOverflow examples from the notebook use the same patience grid as the comparison sweep:

```bash
python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex gmr_fiarse -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 70000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0
python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -num_clients 10 -sample_client high -patience 10 -niid
python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl -num_clients 10 -sample_client low -patience 5 -niid
python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fjord -num_clients 10 -sample_client low -patience 5 -niid
```

The notebook-based StackOverflow comparison sweep uses the following `pr_fl` patience settings:

- IID:
  - high: `15`
  - medium: `20`
  - low: `8`
- Non-IID:
  - high: `10`
  - medium: `10`
  - low: `20`

As above, the other baselines use the explicit experiment names from the notebook cells, and `pr_fl` is the one that varies the patience value directly.

## Notes

- This release is intended to reproduce the public code, experiment setup, and sharable figures behind FedGMR.
- If you want a narrower release, edit `make_public_release.py` and adjust the include/exclude rules.
"""


def matches_any(path_str: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path_str, pat) for pat in patterns)


def should_skip(path: Path, src_root: Path) -> bool:
    rel = path.relative_to(src_root).as_posix()

    if path.is_dir() and path.name in EXCLUDE_DIRS:
        return True
    if path.is_file() and matches_any(path.name, EXCLUDE_FILE_PATTERNS):
        return True
    if matches_any(rel, EXCLUDE_PATH_PATTERNS):
        return True
    return False


def should_include_top_level(path: Path, src_root: Path) -> bool:
    rel_parts = path.relative_to(src_root).parts
    if not rel_parts:
        return True
    if len(rel_parts) == 1 and path.is_file() and path.suffix == ".ipynb":
        return True
    return rel_parts[0] in TOP_LEVEL_INCLUDE


def copy_tree(src_root: Path, dst_root: Path) -> None:
    datasets_src = src_root / "datasets"
    datasets_dst = dst_root / "datasets"
    if datasets_src.exists():
        if datasets_dst.exists() or datasets_dst.is_symlink():
            if datasets_dst.is_dir() and not datasets_dst.is_symlink():
                shutil.rmtree(datasets_dst)
            else:
                datasets_dst.unlink()
        datasets_dst.symlink_to(datasets_src)

    for root, dirnames, filenames in os.walk(src_root):
        root_path = Path(root)

        dirnames[:] = [
            d for d in dirnames
            if should_include_top_level(root_path / d, src_root)
            and not should_skip(root_path / d, src_root)
        ]

        if not should_include_top_level(root_path, src_root):
            continue
        if should_skip(root_path, src_root):
            continue

        rel_root = root_path.relative_to(src_root)
        target_root = dst_root / rel_root
        target_root.mkdir(parents=True, exist_ok=True)

        for fname in filenames:
            src_file = root_path / fname
            if not should_include_top_level(src_file, src_root):
                continue
            if should_skip(src_file, src_root):
                continue
            dst_file = target_root / fname
            shutil.copy2(src_file, dst_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a public code-only release copy of this project."
    )
    parser.add_argument(
        "--src",
        default=".",
        help="Source project directory. Default: current directory.",
    )
    parser.add_argument(
        "--dst",
        default="../AGMR-main-public",
        help="Destination directory for the public release copy.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the destination first if it already exists.",
    )
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if dst_root.exists():
        if not args.force:
            raise SystemExit(
                f"Destination already exists: {dst_root}\n"
                "Re-run with --force to replace it."
            )
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)
    copy_tree(src_root, dst_root)

    readme = dst_root / "README.md"
    readme.write_text(PUBLIC_README, encoding="utf-8")

    print(f"Public release created at: {dst_root}")


if __name__ == "__main__":
    main()
