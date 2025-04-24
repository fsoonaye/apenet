# setup.py
from setuptools import setup, find_packages

setup(
    name="apenet",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0"
    ],
    author="fsoonaye",
    description="A minimalist deep learning library built using NumPy arrays",
    keywords="deep learning, neural networks, numpy",
    python_requires=">=3.6",
)
