# Apenet

Apenet is a minimalist machine learning library built using only NumPy.

The multi-layer perceptron was my first project - I built it to understand how neural networks work from the inside, believing you don't truly understand something unless you build it yourself, and to practice Python OOP. Later, for my research internship, I implemented custom decision trees and random forests to have full control over simulations and extract any statistics needed for my research.

The library is intended for pedagogical and educational purposes and is not recommended for production use. It aims to be human-readable, unlike larger ML libraries, and uses NumPy-style docstrings.

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
