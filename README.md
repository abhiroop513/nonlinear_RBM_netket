
# Project Title

This program designed to create and train a Restricted Boltzmann Machine (RBM) with nonlinear energy function on the Heisenberg model within the framework of the NetKet package (Vicentini et al. , SciPost Phys. Codebases 7 (2022), https://scipost.org/10.21468/SciPostPhysCodeb.7) and JAX and Flax Libraries.

## Key Components

1. **nrbm.py**: This file contains the class of the Restricted Boltzmann Machine (RBM) with nonlinear energy function (power series).
2. **che_rbm.py**: This file contains the class  of the Restricted Boltzmann Machine (RBM) with nonlinear energy function as a Chebyshev expansion.
3. **heisenberg_general.py**: This file contains the class of the Heisenberg model for a general spin degree of freedom $S$ with multiplicity $2S+1$.
4. **conserved_spin_flip.py**: This contains a class with effecient jit-compilable functions  within the JAX and Flax framework to implement a sampling technique, which preserves the total z-component of the spin.
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

