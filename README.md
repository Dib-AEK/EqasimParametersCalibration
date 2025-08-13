# EqasimParametersCalibration

This repository provides tools and scripts for calibrating **mode choice parameters** in **MATSim**, particularly for the *discrete mode choice* extension of MATSim.

It includes modules for:

* Loss calculation
* Mode share analysis
* Optimization algorithms
* Momentum and decay handling
* Network operations
* Utility functions

The goal is to enable **automatic scenario calibration in a single simulation run**, removing the need for manual tuning through multiple simulation iterations.

---

## Repository Structure

**`run.py`**
Main entry point for running calibration workflows.

**`Loss/`**
Modules for calculating and managing loss functions.

**`modeShares/`**
Scripts for analyzing and computing mode shares.

**`MomentumAndDecay/`**
Implements momentum and decay strategies for optimization.

**`Network/`**
Network-related code, including routing and speed calculations *(currently unused)*.

**`Optimizer/`**
Contains various optimization algorithms and factory methods.

**`Selector/`**
Logic for selecting among alternatives in mode choice models.

**`Utilities/`**
Utility classes for different transportation modes and parameter management.

**`utils/`**
General-purpose functions and the CLI interface (`cli.py`).

---

## Key Features

### Flexible Calibration

* Supports multiple optimization algorithms: CMA-ES, Bayesian, GA, and more.
* Integrates momentum strategies such as EMA and Adam.

### Loss Function Customization

* Choose from multiple metrics: MSE, MAE, cosine similarity, KL divergence, etc.

### Mode Share Analysis

* Tools for analyzing simulation outputs and comparing results to observed data.

### Parameter Management

* YAML-based configuration for parameter input/output and bounds.

### Command-Line Interface

* `cli.py` provides a CLI for running calibrations with fully customizable options.

---

## Getting Started

### 1. Install Dependencies

* Requires **Python 3**
* Install required packages:

```bash
pip install -r requirements.txt
```

### 2. Run Calibration

* Using `run.py`:

```bash
python run.py
```


## Author

**Dib Abdelkader**
