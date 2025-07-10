# Optimizer Playground 

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

A comprehensive benchmarking suite for optimization algorithms, with particular focus on evaluating Muon optimizer performance across various machine learning problems.

## Current Implementations

- **Linear Regression** on synthetic data
  - $Y = XW + \varepsilon$ where $Y \in \mathbb{R}^{N \times D}$, $X \in \mathbb{R}^{N \times (d+1)}$, $W \in \mathbb{R}^{(d+1) \times D}$, $\varepsilon \in \mathbb{R}^{N\times D}$
  - $X \sim$ zero centred distribution (typically normal but also supporting Laplace and Uniform), $\varepsilon\sim$ noise_level* zero centred distribution 

- **Logistic Regression** on synthetic data 

- An optimizer comparison suite with a custom implementation of Muon (https://github.com/KellerJordan/Muon) 

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/optimizer_playground.git

# Install dependencies
pip install -r requirements.txt

# Run benchmarks
python main.py