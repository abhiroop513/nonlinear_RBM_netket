import jax
from jax import jit, lax, vmap
import jax.numpy as jnp

from flax import struct

from netket.hilbert.random import flip_state

from netket.sampler.rules.base import MetropolisRule
from netket.sampler.metropolis import MetropolisSampler


class ConservedFlip(struct.PyTreeNode, MetropolisRule):

    def transition(rule, sampler, machine, parameters, state, key, σ):
        # Deduce the number of MCMC chains from input shape
        n_chains = σ.shape[0]

        # Load the Hilbert space of the sampler
        hilb = sampler.hilbert

        def cons_flip(index, index0, σ):

           def upflip(index, σ):

                def cond1(state):
                    index, key2 = state
                    return σ[index] == hilb.local_states[hilb.local_size-1]

                def bf1(state):
                    index, key2 = state
                    key2p, junk = jax.random.split(key2)
                    del junk
                    index2 = jax.random.randint(key2p, shape=(1,), minval=0, maxval=hilb.size)
                    del key2
                    return index2[0], key2p

                state = tuple([index, key2])
                indexup, keyx = lax.while_loop(cond_fun=cond1, body_fun=bf1, init_val=state)

                return indexup

           def downflip(index, indexup, σ):

               def cond2(state):
                  index, key4 = state
                  ind = del_array[index]
                  return σ[ind] == hilb.local_states[0]

               def bf2(state):
                 index, key4 = state
                 key4p, junk = jax.random.split(key4)
                 del junk
                 arr = jnp.arange(len(del_array), dtype=index.dtype)
                 index3 = jax.random.choice(key4p, arr)
                 index3 = index3.astype(index.dtype)
                 del key4
                 return index3, key4p

               res = jnp.arange(len(σ))
               resA = res[:-1]
               resB = res[1:]
               del_array = jnp.where(resA < indexup, resA, resB)

               state = tuple([index, key4])
               indexdn, keyx = lax.while_loop(cond_fun=cond2, body_fun=bf2, init_val=state)

               σp = σ.at[indexup].add(1.0)
               ind2 = del_array[indexdn]
               return σp.at[ind2].add(-1.0)

           indexup = upflip(index, σ)

           σp= downflip(index0, indexup, σ)

           return σp

        key1, key2, key3, key4, key5 = jax.random.split(key,5)


        index = jax.random.randint(key1, shape=(n_chains,),minval=0, maxval=hilb.size)
        index0 = jax.random.randint(key5, shape=(n_chains,),minval=0, maxval=hilb.size-1)

        σp = jax.vmap(cons_flip)(index,index0, σ)

        return σp, None


def MetropolisConservedFlip(hilbert, *args, **kwargs) -> MetropolisSampler:

    return MetropolisSampler(hilbert, ConservedFlip(), *args, **kwargs)
