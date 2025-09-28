
# Nonlinear Restricted Boltzmann Machine (NRBM) for Quantum Many-Body Systems

This program designed to create and train a Restricted Boltzmann Machine (RBM) with nonlinear energy function on the Heisenberg model within the framework of the **NetKet** package (Vicentini et al. , SciPost Phys. Codebases 7 (2022), https://scipost.org/10.21468/SciPostPhysCodeb.7) and JAX and Flax Libraries.

# Nonlinear Restricted Boltzmann Machine (nRBM)

This repository implements a **Restricted Boltzmann Machine (RBM) with a nonlinear energy function**, following the work of  
M. Y. Pei and S. R. Clark, *Entropy* **23** (2021).  
[DOI: 10.3390/e23070879](https://doi.org/10.3390/e23070879)

---

## Nonlinear RBM formulation

The nonlinear RBM is defined as:

$$
F_{\text{nRBM}}(\boldsymbol{\sigma})
= \sum_{\{h_i\}} \exp \Biggl[
    \sum_{n} \sum_{ij} W^{(n)}_{ij}\,\sigma_i^{n}\, h_j
    + \sum_{n=1}^{2S} \sum_i a^{(n)}_i\, \sigma_i^{n}
    + \sum_j h_j b_j
\Biggr]
$$

---

## Spin multiplicity

The number of powers $n$ is equal to the **spin multiplicity minus 1**.  

For example:  
- If $S = \tfrac{3}{2}$, the spin multiplicity is $2S + 1 = 4$,  
  hence $n$ runs up to 4.

## Chebyshev-RBM

Chebyshev polynomials are often better suited for approximating smooth functions than standard power series.  
It has been shown in a previous work (Ref) that using **Chebyshev expansions** in the Jastrow factor improves convergence.

The Chebyshev-RBM is defined as:

$$
F_{\text{che-rbm}}(\boldsymbol{\sigma})
= \sum_{\{h_i\}} \exp \Biggl[
    \sum_{n} \sum_{ij} W^{(n)}_{ij}\,\sigma_i^{n}\, h_j
    + \sum_{n} \sum_i a^{(n)}_i\, \sigma_i^{n}
    + \sum_j h_j b_j
\Biggr]
$$

---

### Remarks

- The replacement of the power expansion with **Chebyshev polynomials** improves the **convergence properties** of the ansatz.  
- Particularly useful when constructing **Jastrow-correlated wavefunctions**.


---


## Key Components

#### This repo contains the following files:

1. **nrbm.py**: This file contains the class of the Restricted Boltzmann Machine (RBM) with nonlinear energy function (power series).
2. **che_rbm.py**: This file contains the class  of the Restricted Boltzmann Machine (RBM) with nonlinear energy function as a Chebyshev expansion.
3. **heisenberg_general.py**: This file contains the class of the Heisenberg model for a general spin degree of freedom $S$ with multiplicity $2S+1$.
4. **conserved_spin_flip.py**: This contains a class with effecient jit-compilable functions  within the JAX and Flax framework to implement a sampling technique, which preserves the total z-component of the spin.\\


---
### Algorithm:Spin-Conserved Sampling

The sampling procedure works as follows:

i)  **Pick a random spin site** $i$.  <br>
ii) Propose an update:
   $`
   S^z_i \to S^z_i + \delta,
   \quad \delta \in \{+1, -1\}.
   `$  <br>
iii) Check if the update is allowed:  If $S^z_i + \delta$ lies outside the valid range $\{-S, -S+1, \dots, S\}$, reject the move. <br>
iv) **Simultaneously select another random spin site** $j \neq i$ and propose:
   $S^z_j \to  S^z_j - \delta.$ <br>
    Again, this is only valid if $S^z_j - \delta$ stays within the allowed spin range.
v) If both updates are valid, perform the move.  <br>
   This ensures that:
   $\sum_i S^z_i = \text{constant},$
   i.e. the total magnetization is conserved.

---
   
5. **main.py**: This file contains the main code to train and test the RBM on the Heisenberg model.
6. **plot_evol.py**: This file contains the code to plot the evolution of the energy of the system as a function of the number of epochs(optimization steps).

# Requirements:

- Python 3.7+
- NetKet 3.0.0+
- JAX 0.2.17+ (Already installed while installing NetKet)


## Installation

1. Install the NetKet library:
```bash
   pip install netket
   ```
   or you can follow the instructions on the NetKet website: https://netket.readthedocs.io/en/latest/docs/install.html

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/repo-name.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd repo-name
   ```
and just run the main program.

## Usage

Explain how to use your project. Provide examples if possible. For instance:

```bash
python main.py --option value
```

Or provide code snippets demonstrating key functionalities.

