# DP-SPSR reproducibility code

This directory contains the Python source and unit tests for the tree-guided DP-SPSR experiments described in the manuscript. The parent reproducibility package also archives the datasets, experiment results, audit records, and manuscript files; this `code/` directory itself contains source files only.

## Requirements

- Python 3.10 or later
- Dependencies listed in `requirements.txt`
- The ToN-IoT CSV used for the reported experiments

The experiment manifests identify the reported input by SHA-256 `26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974`. The data must be obtained and used under its applicable terms; it is not redistributed here.

## Install and test

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest test_fixed_privacy test_task_aware test_tree_personalization
```

## Run

Pass the dataset location and a new output directory explicitly. The scripts write machine-readable experiment outputs to the requested output directory; archived reference outputs are stored under the parent package's `experiments/` directory.

Development-set repeated evaluation:

```powershell
python run_tree_fixed_protocol.py --data "D:\path\to\train_test_network.csv" --output "D:\path\to\results_dev" --grouped-input-holdout
```

Grouped-input reserved-set evaluation:

```powershell
python run_tree_holdout_eval.py --data "D:\path\to\train_test_network.csv" --output "D:\path\to\results_holdout" --grouped-input-holdout
```

Grouped-holdout comparison with classical feature-selection references:

```powershell
python run_grouped_horizontal_comparison.py `
  --data "D:\path\to\train_test_network.csv" `
  --tree-results "D:\path\to\holdout_grouped_input_20261025" `
  --output "D:\path\to\results_horizontal"
```

The added Local-MI, Local-F, Local-MI-Corr, random-K, centralized mRMR-MID,
and Fed-FeRe-inspired chi-square references are non-private. The chi-square
method is a feature-reduction adaptation, not an end-to-end Fed-FeRe
reproduction. `combined_summary.csv` and `paired_macro_f1_differences.csv`
include the tree methods from the supplied holdout result directory.

Grouped-holdout component ablation and correlated-feature pruning:

```powershell
python run_grouped_component_ablation.py `
  --data "D:\path\to\train_test_network.csv" `
  --output "D:\path\to\results_ablation"
```

This ablation reuses the frozen grouped holdout protocol and separates branch
personalization, conditional task selection, tree grouping, cross-client
sharing, DP noise, and fit-fold correlation pruning. The non-private no-noise
variant is a utility reference and does not satisfy the stated DP condition.

Each output path must not already exist. The reserved-set script uses the development partition for client construction, private structure release, tree recovery, feature selection, and classifier fitting; the reserved partition is used only for final evaluation.

## Contents

- `run_tree_fixed_protocol.py`: repeated development evaluation
- `run_tree_holdout_eval.py`: reserved-set evaluation
- `run_grouped_horizontal_comparison.py`: aligned classical-baseline comparison
- `run_grouped_component_ablation.py`: aligned component and feature-redundancy ablation
- `horizontal_baselines.py`: mRMR and federated chi-square selector helpers
- `run_fixed_protocol_pilot.py`: shared evaluation and comparator routines
- `preprocess_real_e6.py`: fixed feature map and client partition utilities
- `dpspsr/`: privacy release, task-aware selection, and tree recovery implementations
- `test_*.py`: focused tests for privacy, task-aware selection, and tree recovery

The experiment scripts use fixed seeds for reproducible research runs. These seeds are for simulation and must not be reused as randomness for deployment privacy releases.
