from typing import List, Optional, Sequence, Union
from numba import jit

import numpy as np
import jax.numpy as jnp
import math

from netket.graph import AbstractGraph, Graph
from netket.hilbert import AbstractHilbert, Fock
from netket.utils.types import DType

from netket.operator import spin
from netket.operator._local_operator import LocalOperator
from netket.operator._graph_operator import GraphOperator
from netket.operator._discrete_operator import DiscreteOperator
#from netket.operator._local_operator_helpers import _dtype

class Heisenberg_general(GraphOperator):
    r"""
    The Heisenberg hamiltonian on a lattice.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        J: Union[float, Sequence[float]] = 1.0,
        sign_rule=None,
        *,
        acting_on_subspace: Union[List[int], int] = None,
    ):
        """
        Constructs an Heisenberg operator given a hilbert space and a graph providing the
        connectivity of the lattice.

        Args:
            hilbert: Hilbert space the operator acts on.
            graph: The graph upon which this hamiltonian is defined.
            J: The strength of the coupling. Default is 1.
               Can pass a sequence of coupling strengths with coloured graphs:
               edges of colour n will have coupling strength J[n]
            sign_rule: If True, Marshal's sign rule will be used. On a bipartite
                lattice, this corresponds to a basis change flipping the Sz direction
                at every odd site of the lattice. For non-bipartite lattices, the
                sign rule cannot be applied. Defaults to True if the lattice is
                bipartite, False otherwise.
                If a sequence of coupling strengths is passed, defaults to False
                and a matching sequence of sign_rule must be specified to override it
            acting_on_subspace: Specifies the mapping between nodes of the graph and
                Hilbert space sites, so that graph node :code:`i ∈ [0, ..., graph.n_nodes - 1]`,
                corresponds to :code:`acting_on_subspace[i] ∈ [0, ..., hilbert.n_sites]`.
                Must be a list of length `graph.n_nodes`. Passing a single integer :code:`start`
                is equivalent to :code:`[start, ..., start + graph.n_nodes - 1]`.

        Examples:
         Constructs a ``Heisenberg`` operator for a 1D system.

            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=10, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=1.5, total_sz=0, N=g.n_nodes)
            >>> op = Heisenberg_general(hilbert=hi, graph=g)
            >>> print(op)
            Heisenberg_general(J=1.0, sign_rule=True; dim=20)
        """
        if isinstance(J, Sequence):
            # check that the number of Js matches the number of colours
            assert len(J) == max(graph.edge_colors) + 1

            if sign_rule is None:
                sign_rule = [False] * len(J)
            else:
                assert len(sign_rule) == len(J)
                for i in range(len(J)):
                    subgraph = Graph(edges=graph.edges(filter_color=i))
                    if sign_rule[i] and not subgraph.is_bipartite():
                        raise ValueError(
                            "sign_rule=True specified for a non-bipartite lattice"
                        )
        else:
            if sign_rule is None:
                sign_rule = graph.is_bipartite()
            elif sign_rule and not graph.is_bipartite():
                raise ValueError("sign_rule=True specified for a non-bipartite lattice")

        self._J = J
        self._sign_rule = sign_rule


        def generate_spin_matrices(S):
            spin_multiplicity = int(2*S + 1)

            # Initialize matrices
            Sx = np.zeros((spin_multiplicity, spin_multiplicity), dtype=complex)
            Sy = np.zeros((spin_multiplicity, spin_multiplicity), dtype=complex)
            Sz = np.zeros((spin_multiplicity, spin_multiplicity), dtype=complex)

            #szcom = np.linspace(S, -S, spin_multiplicity)
            szcom = np.linspace(S, -S+1, int(2*S))
            # Generate Sx and Sy matrices
            #for m in range(S, -S, -1):
            for m in szcom:
                coeff = np.sqrt(S * (S + 1) - m * (m - 1))
                Sx[round(S - m), round(S - m + 1)] = coeff / 2
                Sx[round(S - m + 1), round(S - m)] = coeff / 2
                Sy[round(S - m), round(S - m + 1)] = -1j * coeff / 2
                Sy[round(S - m + 1), round(S - m)] = 1j * coeff / 2
                
            szcom = np.linspace(S, -S, int(2*S+1))
            # Generate Sz matrix
            for m in szcom:
                Sz[round(S - m), round(S - m)] = m

            return Sx, Sy, Sz

        # Now Generate spin matrices for S=3/2 (spin multiplicity 4)
        S = 1.5
        sx, sy, sz = generate_spin_matrices(S)

        sp = sx + 1.0j*sy
        sm = sx - 1.0j*sy

        sz_sz = np.kron(sz, sz)
        exchange = np.kron(sp,sm) + np.kron(sm,sp)

        if isinstance(J, Sequence):
            bond_ops = [
                J[i] * (sz_sz - 0.5*exchange if sign_rule[i] else sz_sz + 0.5*exchange)
                for i in range(len(J))
            ]
            bond_ops_colors = list(range(len(J)))
        else:
            bond_ops = [J * (sz_sz - 0.5*exchange if sign_rule else sz_sz + 0.5*exchange)]
            bond_ops_colors = []

        super().__init__(
            hilbert,
            graph,
            bond_ops=bond_ops,
            bond_ops_colors=bond_ops_colors,
            acting_on_subspace=acting_on_subspace,
        )

    @property
    def J(self) -> float:
        """The coupling strength."""
        return self._J

    @property
    def uses_sign_rule(self):
        return self._sign_rule

    def __repr__(self):
        return f"Heisenberg_general(J={self._J}, sign_rule={self._sign_rule}; dim={self.hilbert.size})"
