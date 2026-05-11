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

This directory is a public, code-oriented release of the FedGMR project.

## Included

- core training and aggregation code
- experiment scripts and configuration files
- selected visualization assets under `figure/`
- root-level notebook files (`*.ipynb`)
- a symlink to the shared `datasets/` directory
- environment helper script `setenv.sh`
- control modules under `control/`
- utility modules under `utils/`

## Excluded

- paper source files (`.tex`, `.bib`, style files, submission-only assets)
- rebuttal documents and confidential review comments
- large raw result directories and other non-public release artifacts

## Running the Code

Before running any experiment, source the environment helper:

```bash
source setenv.sh
```

The public release expects `datasets/` to be a symlink to the shared dataset storage. This keeps the release lightweight while reusing the prepared datasets outside the repository.

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

The notebook comparisons sweep patience over `3`, `5`, `10`, and `15`; the shell scripts also use `3`, `5`, and `10` depending on the recovery mode.

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

The shell scripts use patience `5` for `pr_fl` / `fed_asyn` examples and patience `10` for the `heterofl` example.

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

The notebook comparison sweeps patience over `5`, `10`, and `15`; the time-trigger example in the shell script uses patience `7`.

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

- This release is intended to reproduce code structure, experiment setup, and figures that are safe to share.
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
