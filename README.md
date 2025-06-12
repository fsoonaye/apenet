# Apenet

Apenet is a minimalist machine learning library built using only NumPy.

The multi-layer perceptron was my first project - I built it to understand how neural networks work from the inside. Later, I had to implement custom decision trees and random forests to have full control over simulations and extract any statistics needed for my research. I then decided to merge both projects.

The library is intended for pedagogical and educational purposes and is not recommended for production use. It aims to readable and uses NumPy-style docstrings.


## Features Overview

### Main Components

The library is divided into three main parts:

1. **nn**: Contains modules for neural network models.
2. **rf**: Contains modules for random forest models.
3. **eye**: Contains functions for data visualization.

### Neural Networks (nn)
- **Models**:
  - Multi-Layer Perceptron

- **Loss Functions**: 
  - CrossEntropyLoss for classification
  - MSELoss for regression

- **Optimizers**: 
  - SGD with momentum
  - Adam

- **Activation Functions**:
  - ReLU
  - Sigmoid
  - Tanh
  - Softmax

- **Weight Initializers**:
  - Xavier initialization
  - He initialization
  - Normal initialization
  - Zeros initialization


### Random Forests (rf)
- **Decision Trees**:
  - Classification trees with Gini impurity
  - Regression trees with variance reduction
  - Configurable stopping criterias
  - MDI computation

- **Random Forests**:
  - Regression with averaging
  - Classification with majority voting
  - Bootstrap sampling
  - Feature importance calculation

### Visualization (eye)
- **History Visualization**:
  - plot_history for tracking training metrics (accuracy/loss vs epoch)

- **Tree Visualization**:
  - plot_tree_structure for visualizing decision tree structure
  - plot_decision_regions for visualizing model decision regions and splits

- **Data Visualization**:
  - plot_confusion_matrix for classification evaluation

### Utilities
- **Metrics**:
  - Accuracy for classification
  - Mean squared error for regression

- **Data Processing**:
  - Bootstrap sampling
  - Standardize

### Directory Structure

```
apenet/
├── eye
│   ├── data.py
│   ├── history.py
│   └── tree.py
├── interfaces.py
├── nn/
│   ├── activ.py
│   ├── init.py
│   ├── layer.py
│   ├── loss.py
│   ├── models
│   │   ├── mlp.py
│   │   └── sequential.py
│   ├── optim.py
│   └── utils.py
├── rf/
│   ├── forest.py
│   └── tree.py
└── utils/
    ├── data.py
    └── metrics.py

6 directories, 16 files
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
