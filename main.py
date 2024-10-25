# This is the main program for running the RBM model on the Heisenberg model.
import json

from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.exact import lanczos_ed

import flax
from flax.core import frozen_dict
import flax.linen as nn
import jax.numpy as jnp
import jax

import netket as nk
from netket import models
from netket import operator
from netket import sampler
from netket import optimizer
from netket import vqs
from netket.driver import VMC
import netket.jax as nkjax
#import msgpack

import numpy as np
import matplotlib.pyplot as plt
import time

from collections import Counter


from heisenberg_general import Heisenberg_general
from conserved_spin_flip import MetropolisConservedFlip
from nrbm3 import RBMCube
#from che_rbm3 import RBM_Chebyshev_3

#a = np.array([0., 1.0, 2.0])

#if __name__ == '__main__':
#print(a)
L = 6
g = Hypercube(length=L, n_dim=1, pbc=True)

print(g)

# Define the Hilbert space based on this graph
# We impose to have a fixed total magnetization of zero
hi = Spin(s=1.5, total_sz=0, N=g.n_nodes)

# calling the Heisenberg Hamiltonian
ha = Heisenberg_general(hilbert=hi, graph=g)
# The nonlinear Restricted Boltzmann Machine (RBM) model
ma = RBMCube(num_neurons=L , alpha=2.0, weight_init=nn.initializers.normal(stddev=0.0001))
#ma = models.RBM(alpha=1)
print(ma)

#sa = sampler.MetropolisLocal(hilbert=hi)
#The sampler
sa = MetropolisConservedFlip(hilbert=hi)
#exit()
# Optimizer
op = optimizer.Sgd(learning_rate=0.001)

# Stochastic Reconfiguration
sr = optimizer.SR(diag_shift=0.001)

# The variational state

vs = vqs.MCState(sa, ma, n_samples=500)

#sf = []
#sites = []
#structure_factor = operator.LocalOperator(hi, dtype=complex)
#for i in range(0, L):
#    for j in range(0, L):
#        structure_factor += (operator.spin.sigmaz(hi, i)*operator.spin.sigmaz(hi, j))*((-1)**(i-j))/L

#global param_list
#param_list = []

# The ground-state optimization loop
gs = VMC(
    hamiltonian=ha,
    optimizer=op,
    preconditioner=sr,
    variational_state=vs
    )
#append_param=False

start = time.time()
#gs.run(300, out='Jastrow')
log = nk.logging.JsonLog('RBM')
gs.run(out=log, n_iter=300)

end = time.time()

print('### RBM calculation')
print('Has',nkjax.tree_size(vs.parameters),'parameters')
print('The RBM calculation took',end-start,'seconds')

