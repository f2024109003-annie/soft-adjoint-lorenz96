# Soft Adjoint Operators for the Lorenz–96 Model

This repository provides a **minimal, reproducible Python/JAX implementation**
of the *soft adjoint framework* proposed in the accompanying research paper.

The code demonstrates how ensemble-based uncertainty can be aggregated
into a single soft adjoint gradient computation for a chaotic dynamical system.

---

##  Purpose of this Repository

- Verify numerical claims made in the paper
- Provide transparent and reproducible code
- Demonstrate the soft adjoint concept on the Lorenz–96 system

This repository focuses **only on the Lorenz–96 benchmark**.
Quadrotor experiments are provided in a separate repository.

---

##  Repository Contents

- `soft_adjoint_lorenz96.py`  
  Python/JAX implementation of:
  - Lorenz–96 dynamics
  - Ensemble (soft set) forcing
  - Soft adjoint gradient computation

---

##  Model Description

The Lorenz–96 system is a standard low-order atmospheric proxy model used to
test scalability and robustness of data assimilation and sensitivity methods
in chaotic regimes.

Soft adjoints aggregate gradients over an ensemble of forcings,
avoiding multiple independent adjoint solves.

---

##  Relation to Paper Results

This code supports:
- RMSE values reported for the Lorenz–96 experiment
- Linear ensemble scaling behavior
- Timing comparisons with classical and ensemble adjoint baselines

Exact numerical values may vary slightly depending on hardware and random
initialization.

---

##  Data and Forcing

The ensemble forcing used here is **synthetic** and designed to emulate
ECMWF ERA5 ensemble variability.

Raw ERA5 NetCDF files are not included due to size and licensing constraints.
Users may download ERA5 data directly from:

https://cds.climate.copernicus.eu

---

##  How to Run

```bash
python soft_adjoint_lorenz96.py
