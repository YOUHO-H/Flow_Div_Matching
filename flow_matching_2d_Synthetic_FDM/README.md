# Flow Matching with Flow Divergence for Synthetic density estimation

## Installation

This repository requires Python 3.9 and Pytorch 2.1 or greater. To install the latest version run:
```
pip install flow_matching
```

## Repository structure

The core and example folders are structured in the following way:
```bash
.
├── flow_matching                  # Core library
│   ├── loss                       # Loss functions
│   │   └── ...
│   ├── path                       # Path and schedulers
│   │   ├── ...
│   │   └── scheduler              # Schedulers and transformations
│   │       └── ...
│   ├── solver                     # Solvers for continuous and discrete flows
│   │   └── ...
│   └── utils
│       └── ...
└── main
```

## Development

To create a conda environment with all required dependencies, run:
```
conda env create -f environment.yml
conda activate flow_matching
```

Install pre-commit hook. This will ensure that all linting is done on each commit
```
pre-commit install
```

Install the `flow_matching_2d_Synthetic_density_estimation` package in an editable mode:
```
pip install -e .
```
## run experiment
```
python main_fm_2d.py
```
