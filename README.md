# Constraint-Aware Adversarial Evaluation of DL-NIDS

Anonymized artifact for the paper *"Robust Network Intrusion Detection Against
Constraint-Aware Adversarial Evasion."* This repository reproduces all
experiments, tables, and figures.

## Overview

The pipeline evaluates the adversarial robustness of deep-learning network
intrusion detectors under **feature-domain constraint validity**: adversarial
examples are projected back to the space of constraint-valid feature vectors
(per-feature bounds, integrality of count-based features, and immutability of
attacker-uncontrollable fields). It contrasts the constraint-valid threat with
the standard unconstrained one, reports the **validity rate** of unconstrained
attacks, and defends via constrained adversarial training.

## Requirements

```
pip install -r requirements.txt
```

Python 3.10 and a CUDA-capable GPU are recommended. CPU works but is slow.

## Dataset

We use the **corrected CIC-IDS2017** dataset. It is not redistributed here;
download the corrected CSVs from the sources cited in the paper and place them
in `./data/`.

## Preprocessing note

`preprocess.py` excludes a row-identifier column (`id`) and a non-semantic
bookkeeping column (`Attempted Category`) that do not correspond to flow
measurements, leaving **84 features** (14 immutable, 63 integer-valued). Set
`EXTRA_HYGIENE = True` inside `clean()` to additionally drop the port columns.

## Reproducing everything (one command)

```
python run_all.py
```

`run_all.py` regenerates data for both label settings, trains all victims and
defenses (MLP and 1D-CNN), and runs the full study (eps sweep, ablation,
per-class, macro-F1, violation breakdown, targeted benign-class attack). It
writes `outputs/experiments/summary_for_paper.txt` plus CSVs and figures.

To reproduce the additional revision experiments:

```
python run_adaptive_and_seeds.py   # multi-restart adaptive attack + multi-seed variance
python revision_extras.py          # macro-F1, 4-way ablation, feature lists, class distribution
python violation_breakdown.py      # per-constraint-type violation breakdown of the 0% result
python targeted_attack.py          # attack-to-benign success rate
```

Or run the core pipeline step by step:

```
python preprocess.py --data_dir ./data --out_dir ./outputs --binary   # or omit --binary for multiclass
python model.py --out_dir ./outputs --epochs 30
python defense_eval.py --out_dir ./outputs --epochs 20 --steps 20
python run_experiments.py
python run_arch_and_baseline.py
python make_figures.py
```

## File guide

| File | Role |
|------|------|
| `preprocess.py` | Load/clean CIC-IDS2017 (84-feature spec), split, build constraint spec |
| `model.py` | MLP victim detector; training and clean-baseline evaluation |
| `model_cnn.py` | 1D-CNN victim (drop-in; same attack surface as the MLP) |
| `attack.py` | Constrained vs. unconstrained PGD; projection + validity rate |
| `defense_eval.py` | Constrained adversarial training + single-config results |
| `run_experiments.py` | Sweep / ablation / per-class study, both settings |
| `run_arch_and_baseline.py` | Architecture comparison + training-type baseline |
| `run_all.py` | One-shot orchestrator for the whole paper |
| `run_adaptive_and_seeds.py` | Multi-restart adaptive attack + multi-seed variance |
| `revision_extras.py` | Macro-F1, four-variant ablation, feature/class listings |
| `violation_breakdown.py` | Which constraint types unconstrained attacks violate |
| `targeted_attack.py` | Targeted benign-class evasion + success rate |
| `make_figures.py` | Figures and LaTeX tables from the generated CSVs |

## Ethics

All attacks are used solely to evaluate and improve defensive robustness in an
offline setting on a public benchmark, as described in the paper.
