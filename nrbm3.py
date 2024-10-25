from typing import Union, Any
import jax
import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import normal
from jax import lax
from flax.linen.dtypes import promote_dtype
import numpy as np

from netket.utils import HashableArray #deprecate_dtype
from netket.utils.types import NNInitFunc
from netket import nn as nknn
from typing import Any, Callable, Sequence, Tuple, Optional

default_kernel_init = normal(stddev=0.01)

PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
Dtype = Any


class RBMCube(nn.Module):
    num_neurons: int = 1
    weight_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    alpha: Union[float, int] = 1
    #features : int = int(alpha * num_neurons)
    use_visible_bias: bool = True
    #kernel_init: NNInitFunc = default_kernel_init
    visible_bias_init: NNInitFunc = default_kernel_init
    param_dtype: Any = np.float64
    precision: PrecisionLike = None
    dtype: Optional[Dtype] = None
    activation: Any = nknn.log_cosh
    #features: 0.5*x.shape[-1]
    #n_classes:int = 3
    print("default_kernel_init  ", default_kernel_init)

    @property
    def features(self):
        return int(self.alpha * self.num_neurons)
    @nn.compact
    def __call__(self, x):
  #==============================================================
        ## Convert Normal RBM to One HOT ENCODING

        #x_oh = jax.nn.one_hot(x, self.n_classes)

        #kernel = self.param('kernel',  # parametar name (as it will appear in the FrozenDict)
        #            self.weight_init,  # initialization function, RNG passed implicitly through init fn
        #            (self.features, x.shape[-1], self.n_classes))

  #===============================================================

        kernel = self.param('kernel',  # parametar name (as it will appear in the FrozenDict)
                self.weight_init,  # initialization function, RNG passed implicitly through init fn
                (x.shape[-1],  self.features))
        kernel2 = self.param('kernel2',  # parametar name (as it will appear in the FrozenDict)
                self.weight_init,  # initialization function, RNG passed implicitly through init fn
                (x.shape[-1], self.features))
        kernel3 = self.param('kernel3',  # parametar name (as it will appear in the FrozenDict)
                self.weight_init,  # initialization function, RNG passed implicitly through init fn
                (x.shape[-1], self.features))

        #print("X SHAPE ", x.shape[-1])
               #(x.shape[-1], self.num_neurons))  # shape info
        bias = self.param('bias', self.bias_init, self.features)


        #return jnp.dot(x, weight)  + bias

        x, kernel, kernel2, kernel3, bias = promote_dtype(x, kernel, kernel2, kernel3, bias, dtype=self.dtype)
#       y = lax.dot_general(x, kernel,
#                       (((x.ndim - 1,), (0,)), ((), ())),
#                       precision=self.precision)
        x = x/1.5
        y = lax.dot(x, kernel,precision=self.precision) + lax.dot(lax.square(x), kernel2,
                        precision=self.precision) + lax.dot(lax.pow(x,3.0), kernel3,precision=self.precision)
        y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        y = self.activation(y)
        y = jnp.sum(y, axis=-1)

        if self.use_visible_bias:
          v_bias = self.param(
              "visible_bias",
               self.visible_bias_init,
               (x.shape[-1],),
               self.param_dtype,
            )
          v_bias_2 = self.param(
              "visible_bias_2",
               self.visible_bias_init,
               (x.shape[-1],),
               self.param_dtype,
            )
          v_bias_3 = self.param(
              "visible_bias_3",
               self.visible_bias_init,
               (x.shape[-1],),
               self.param_dtype,
            )
          out_bias = jnp.dot(x, v_bias) + jnp.dot(jnp.square(x), v_bias_2) \
                     + jnp.dot(jnp.power(x,3.0), v_bias_3)
          #out_bias = jnp.dot(x, v_bias)

          #return jnp.add(y,bias)
          #return jnp.dot(x, kernel) + bias + out_bias
          return y + out_bias
        else:
          return y