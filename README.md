# FedGMR Public Release

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

### 1. Stage-dependent benefit of model density

This figure illustrates the central empirical observation behind FedGMR: smaller sub-models improve faster early, while larger-capacity models become more beneficial later.

![Learning-rate analysis](FedGMR/learning_rate_different_density_models_strip.png)

### 2. Cross-method gains from GMR

Applying GMR to multiple MHFL baselines consistently improves performance. This supports that the main contribution is the **restoration mechanism itself**, rather than one specific pruning rule or one specific base method.

![Cross-method comparison](FedGMR/GMR_on_other_MHFL_methods.png)

### 3. Aggregation and ablation analysis

The ablation results highlight that restoration and aggregation are coupled: GMR improves late-stage usefulness, while mask-aware aggregation helps maintain stability during restoration.

![Ablation analysis](FedGMR/Ablation_method.png)

### 4. Robustness to restoration timing

FedGMR does not depend on one exact restoration trigger. A fixed-time restoration variant remains effective across datasets, although adaptive triggering can further improve some settings.

![Fixed-time restoration](FedGMR/Fixed_restoration_timeing_plot.png)

## Representative Results

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
