# Constraint-Aware Adversarial Evaluation of DL-NIDS

Anonymized artifact for the paper *"Robust Network Intrusion Detection Against
Constraint-Aware Adversarial Evasion."* This repository contains the code to
reproduce all experiments, figures, and tables.

## Overview

The pipeline evaluates the adversarial robustness of deep-learning network
intrusion detectors under a **realizability constraint**: adversarial examples
are projected back to the space of valid network flows (respecting per-feature
bounds, integrality of count-based features, and immutability of
attacker-uncontrollable fields). It compares the realizable (constrained) threat
against the standard unconstrained one, reports the **validity rate** of
unconstrained attacks, and defends via constrained adversarial training.

## Requirements

```
pip install -r requirements.txt
```

Python 3.10 and a CUDA-capable GPU are recommended (experiments were run on a
single 48 GB GPU). CPU execution works but is slow.

## Dataset

We use the **corrected CIC-IDS2017** dataset. The dataset is not redistributed
here; download the corrected CSVs from the sources cited in the paper (the
WTMC-2021 / CNS-2022 corrected releases) and place them in `./data/`.

## Directory layout

```
./data/                 # corrected CIC-IDS2017 CSVs (user-provided)
./outputs/              # generated: data.npz, constraint_spec.json, models
./outputs/experiments/  # generated: CSVs, per-experiment models
./outputs/experiments/figures/  # generated: PDF figures + LaTeX tables
```

## Reproducing the results

Run the scripts in order. All paths default to `./data` and `./outputs`
(relative to the repository root); override with `--data_dir` / `--out_dir`
where applicable.

```bash
# 1. Preprocess: clean, split, build the constraint specification.
#    Use --binary for the binary setting; omit it for multiclass.
python preprocess.py --data_dir ./data --out_dir ./outputs --binary

# 2. Train the victim detector and record the clean baseline.
python model.py --out_dir ./outputs --epochs 30

# 3. Single-configuration attack + defense evaluation (eps = 0.3).
python defense_eval.py --out_dir ./outputs --epochs 20 --steps 20

# 4. Full study: eps sweep, ablation, per-class, both label settings.
python run_experiments.py

# 5. Architecture comparison (MLP vs 1D-CNN) + training-type baseline
#    (constrained vs unconstrained adversarial training).
python run_arch_and_baseline.py

```

## File guide

| File | Role |
|------|------|
| `preprocess.py` | Load/clean CIC-IDS2017, split, build the constraint spec |
| `model.py` | MLP victim detector; training and clean-baseline evaluation |
| `model_cnn.py` | 1D-CNN victim (drop-in; same attack surface as the MLP) |
| `attack.py` | Constrained vs. unconstrained PGD; projection + validity rate |
| `defense_eval.py` | Constrained adversarial training + single-config results |
| `run_experiments.py` | Full sweep/ablation/per-class study, both settings |
| `run_arch_and_baseline.py` | Architecture comparison + training-type baseline |

