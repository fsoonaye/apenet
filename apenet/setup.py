# setup.py
from setuptools import setup, find_packages

setup(
    name="apenet",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0"
    ],
    extras_require={
        'eye': [
            "matplotlib",
            "pygraphviz",
            "networkx"
        ],
        'rf': [
        ],
        'nn': [
        ]
    },
    author="fsoonaye",
    description="A minimalist machine learning library built using NumPy arrays",
    keywords="deep learning, neural networks, numpy",
    python_requires=">=3.6",
)
