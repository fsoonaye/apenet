# Apenet

Apenet is a minimalist machine learning library built using only NumPy arrays and matrix multiplication.  

Apenet was written to help me grasp key concepts in machine learning and is intended for pedagogical and educational purposes. It is not recommended for heavy or efficiency-critical tasks.  

It aims to be human-readable, unlike larger and more efficient data science/ML libraries such as scikit-learn, NumPy, or PyTorch. Given its direct usage, it uses NumPy-style docstrings.

## Main Components

The library is divided into three main parts:

1. **nn**: Contains modules for neural network models.
2. **rf**: Contains modules for random forest models.
3. **eye**: Contains functions for data visualization.

## Directory Structure

```
apenet
├── eye
│   └── visuals.py
├── interfaces
│   └── base_model.py
├── nn
│   ├── core
│   │   ├── activations.py
│   │   └── layers.py
│   ├── initialize
│   │   └── initializers.py
│   ├── loss
│   │   └── losses.py
│   ├── models
│   │   ├── mlp.py
│   │   └── sequential.py
│   ├── optimizers
│   │   └── optimizers.py
│   └── utils
│       └── helpers.py
├── rf
│   └── rf.py
└── utils
    ├── data.py
    └── metrics.py

12 directories, 14 files
```

## Installation

It is strongly recommended to install Apenet in a virtual environment. Here is how you can set it up using `venv`:

```sh
# Create a virtual environment
python -m venv apenet-env

# Activate the virtual environment
# On Windows
apenet-env\Scripts\activate
# On macOS/Linux
source apenet-env/bin/activate

# Install the library
pip install .
```

## Example Usage

Example usage of the library on toy datasets can be found in the `tests/` directory.
