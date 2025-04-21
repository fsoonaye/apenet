# setup.py
from setuptools import setup, find_packages

setup(
    name="apenet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.0.0"
    ],
    author="fsoonaye",
    description="A minimalist deep learning library built using PyTorch tensors",
    keywords="deep learning, neural networks, pytorch",
    python_requires=">=3.6",
)