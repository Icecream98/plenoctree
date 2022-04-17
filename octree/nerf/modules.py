# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules for NeRF models."""
import functools
from typing import Optional, Tuple

#import torch.nn as nn

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from nerf import types

class SinusoidalEncoder(nn.Module):
  """A vectorized sinusoidal encoding.

  Attributes:
    num_freqs: the number of frequency bands in the encoding.
    min_freq_log2: the log (base 2) of the lower frequency.
    max_freq_log2: the log (base 2) of the upper frequency.
    scale: a scaling factor for the positional encoding.
    use_identity: if True use the identity encoding as well.
  """
  num_freqs: int
  min_freq_log2: int = 0
  max_freq_log2: Optional[int] = None
  scale: float = 1.0
  use_identity: bool = True

  def setup(self):
    if self.max_freq_log2 is None:
      max_freq_log2 = self.num_freqs - 1.0
    else:
      max_freq_log2 = self.max_freq_log2
    self.freq_bands = 2.0**jnp.linspace(self.min_freq_log2,
                                        max_freq_log2,
                                        int(self.num_freqs))

    # (F, 1).
    self.freqs = jnp.reshape(self.freq_bands, (self.num_freqs, 1))

  def __call__(self, x, alpha: Optional[float] = None):
    """A vectorized sinusoidal encoding.

    Args:
      x: the input features to encode.
      alpha: a dummy argument for API compatibility.

    Returns:
      A tensor containing the encoded features.
    """
    # ***************************  original setup BEG  **********************
    # 这样做是纯当函数用了，torch里用flax，不支持保存变量啥的 
    if self.max_freq_log2 is None:
      max_freq_log2 = self.num_freqs - 1.0
    else:
      max_freq_log2 = self.max_freq_log2
    self.freq_bands = 2.0**jnp.linspace(self.min_freq_log2,
                                        max_freq_log2,
                                        int(self.num_freqs))

    # (F, 1).
    self.freqs = jnp.reshape(self.freq_bands, (self.num_freqs, 1))
    # ***************************  original setup END  **********************
    if self.num_freqs == 0:
      return x

    x_expanded = jnp.expand_dims(x, axis=-2)  # (1, C).
    # Will be broadcasted to shape (F, C).
    angles = self.scale * x_expanded * self.freqs

    # The shape of the features is (F, 2, C) so that when we reshape it
    # it matches the ordering of the original NeRF code.
    # Vectorize the computation of the high-frequency (sin, cos) terms.
    # We use the trigonometric identity: cos(x) = sin(x + pi/2)
    features = jnp.stack((angles, angles + jnp.pi / 2), axis=-2)
    features = features.flatten()
    features = jnp.sin(features)

    # Prepend the original signal for the identity.
    if self.use_identity:
      features = jnp.concatenate([x, features], axis=-1)
    return features


# class SinusoidalEncoder(nn.Module):
#   """A vectorized sinusoidal encoding.

#   Attributes:
#     num_freqs: the number of frequency bands in the encoding.
#     min_freq_log2: the log (base 2) of the lower frequency.
#     max_freq_log2: the log (base 2) of the upper frequency.
#     scale: a scaling factor for the positional encoding.
#     use_identity: if True use the identity encoding as well.
#   """
#   def __init__(
#     self,
#     num_freqs: int = 8,
#     min_freq_log2: int = 0,
#     max_freq_log2: Optional[int] = None,
#     scale: float = 1.0,
#     use_identity: bool = True,
#   ):
#     self.num_freqs=8
#     self.min_freq_log2=min_freq_log2
#     self.max_freq_log2=max_freq_log2
#     self.scale=scale
#     self.use_identity=use_identity
    
#     if self.max_freq_log2 is None:
#       max_freq_log2 = self.num_freqs - 1.0
#     else:
#       max_freq_log2 = self.max_freq_log2
#     self.freq_bands = 2.0**np.linspace(self.min_freq_log2,
#                                         max_freq_log2,
#                                         int(self.num_freqs))

#     # (F, 1).
#     self.freqs = np.reshape(self.freq_bands, (self.num_freqs, 1))
  
#   def __call__(self, x, alpha: Optional[float] = None):
#     """A vectorized sinusoidal encoding.

#     Args:
#       x: the input features to encode.
#       alpha: a dummy argument for API compatibility.

#     Returns:
#       A tensor containing the encoded features.
#     """
#     if self.num_freqs == 0:
#       return x

#     x_expanded = np.expand_dims(x, axis=-2)  # (1, C).
#     # Will be broadcasted to shape (F, C).
#     angles = self.scale * x_expanded * self.freqs

#     # The shape of the features is (F, 2, C) so that when we reshape it
#     # it matches the ordering of the original NeRF code.
#     # Vectorize the computation of the high-frequency (sin, cos) terms.
#     # We use the trigonometric identity: cos(x) = sin(x + pi/2)
#     features = np.stack((angles, angles + np.pi / 2), axis=-2)
#     features = features.flatten()
#     features = np.sin(features)

#     # Prepend the original signal for the identity.
#     if self.use_identity:
#       features = np.concatenate([x, features], axis=-1)
#     return features

