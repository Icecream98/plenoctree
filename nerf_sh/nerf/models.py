# coding=utf-8
# Modifications Copyright 2021 The PlenOctree Authors.
# Original Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Different model implementation plus a general port for all the models."""
from typing import Any, Callable
import flax
from flax import linen as nn
from jax import random
import jax.numpy as jnp

from nerf import model_utils
from nerf import utils
from nerf import sh
from nerf import sg
from nerf import modules
from nerf import warping


def get_model(key, args):
    """A helper function that wraps around a 'model zoo'."""
    model_dict = {
        "nerf": construct_nerf,
    }
    return model_dict[args.model](key, args)

# def get_model_state(key, args, restore=True):
#     """
#     Helper for loading model with get_model & creating optimizer &
#     optionally restoring checkpoint to reduce boilerplate
#     """
#     model, variables = get_model(key, args)
#     optimizer = flax.optim.Adam(args.lr_init).create(variables)
#     state = utils.TrainState(
#         optimizer=optimizer,
#         warp_alpha=warp_alpha_sched(0),
#         time_alpha=time_alpha_sched(0))
#     if restore:
#         from flax.training import checkpoints
#         state = checkpoints.restore_checkpoint(args.train_dir, state)
#     return model, state


class NerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    num_coarse_samples: int  # The number of samples for the coarse nerf.
    num_fine_samples: int  # The number of samples for the fine nerf.
    use_viewdirs: bool  # If True, use viewdirs as an input.
    sh_deg: int  # If != -1, use spherical harmonics output up to given degree
    sg_dim: int  # If != -1, use spherical gaussians output of given dimension
    near: float  # The distance to the near plane
    far: float  # The distance to the far plane
    noise_std: float  # The std dev of noise added to raw sigma.
    net_depth: int  # The depth of the first part of MLP.
    net_width: int  # The width of the first part of MLP.
    net_depth_condition: int  # The depth of the second part of MLP.
    net_width_condition: int  # The width of the second part of MLP.
    net_activation: Callable[Ellipsis, Any]  # MLP activation
    skip_layer: int  # How often to add skip connections.
    num_rgb_channels: int  # The number of RGB channels.
    num_sigma_channels: int  # The number of density channels.
    white_bkgd: bool  # If True, use a white background.
    min_deg_point: int  # The minimum degree of positional encoding for positions.
    max_deg_point: int  # The maximum degree of positional encoding for positions.
    deg_view: int  # The degree of positional encoding for viewdirs.
    lindisp: bool  # If True, sample linearly in disparity rather than in depth.
    rgb_activation: Callable[Ellipsis, Any]  # Output RGB activation.
    sigma_activation: Callable[Ellipsis, Any]  # Output sigma activation.
    legacy_posenc_order: bool  # Keep the same ordering as the original tf code.
    num_nerf_point_freqs: int # nerfies config
    num_nerf_viewdir_freqs: int # nerfies config
    use_warp: bool # nerfies config

    def setup(self):
        # Construct the "coarse" MLP. Weird name is for
        # compatibility with 'compact' version
        
        if self.use_warp:
            self.warp_field = self.create_warp_field(self, num_batch_dims=2)
            
        self.MLP_0 = model_utils.MLP(
            net_depth=self.net_depth,
            net_width=self.net_width,
            net_depth_condition=self.net_depth_condition,
            net_width_condition=self.net_width_condition,
            net_activation=self.net_activation,
            skip_layer=self.skip_layer,
            num_rgb_channels=self.num_rgb_channels,
            num_sigma_channels=self.num_sigma_channels,
        )

        # Construct the "fine" MLP.
        self.MLP_1 = model_utils.MLP(
            net_depth=self.net_depth,
            net_width=self.net_width,
            net_depth_condition=self.net_depth_condition,
            net_width_condition=self.net_width_condition,
            net_activation=self.net_activation,
            skip_layer=self.skip_layer,
            num_rgb_channels=self.num_rgb_channels,
            num_sigma_channels=self.num_sigma_channels,
        )

        # Construct global learnable variables for spherical gaussians.
        if self.sg_dim > 0:
            key1, key2 = random.split(random.PRNGKey(0), 2)
            self.sg_lambda = self.variable(
                "params", "sg_lambda",
                lambda x: jnp.ones([x], jnp.float32), self.sg_dim)
            self.sg_mu_spher = self.variable(
                "params", "sg_mu_spher",
                lambda x: jnp.concatenate([
                    random.uniform(key1, [x, 1]) * jnp.pi,  # theta
                    random.uniform(key2, [x, 1]) * jnp.pi * 2,  # phi
                ], axis=-1), self.sg_dim)

        self.point_encoder = model_utils.vmap_module(
            modules.SinusoidalEncoder, num_batch_dims=2)(
                num_freqs=self.num_nerf_point_freqs)
        self.viewdir_encoder = model_utils.vmap_module(
            modules.SinusoidalEncoder, num_batch_dims=1)(
                num_freqs=self.num_nerf_viewdir_freqs)

    def _quick_init(self):
        points = jnp.zeros((1, 1, 3), dtype=jnp.float32)
        # points_enc = model_utils.posenc(
        #     points,
        #     self.min_deg_point,
        #     self.max_deg_point,
        #     self.legacy_posenc_order,
        # )
        points_enc=self.point_encoder(points)
        if self.use_viewdirs:
            viewdirs = jnp.zeros((1, 1, 3), dtype=jnp.float32)
            # viewdirs_enc = model_utils.posenc(
            #     viewdirs,
            #     0,
            #     self.deg_view,
            #     self.legacy_posenc_order,
            # )
            viewdirs_enc=self.viewdir_encoder(viewdirs)
            self.MLP_0(points_enc, viewdirs_enc)
            if self.num_fine_samples > 0:
                self.MLP_1(points_enc, viewdirs_enc)
        else:
            self.MLP_0(points_enc)
            if self.num_fine_samples > 0:
                self.MLP_1(points_enc)

    @staticmethod
    def create_warp_field(model, num_batch_dims):
        return warping.create_warp_field(
            field_type=model.warp_field_type,
            num_freqs=model.num_warp_freqs,
            num_embeddings=model.num_warp_embeddings,
            num_features=model.num_warp_features,
            num_batch_dims=num_batch_dims,
            metadata_encoder_type=model.warp_metadata_encoder_type,
            **model.warp_kwargs)
        
    def eval_points_raw(self, points, viewdirs=None, coarse=False):
        """
        Evaluate at points, returing rgb and sigma.
        If sh_deg >= 0 / sg_dim > 0 then this will return
        spherical harmonic / spherical gaussians / anisotropic spherical gaussians
        coeffs for RGB. Please see eval_points for alternate
        version which always returns RGB.
        Args:
          points: jnp.ndarray [B, 3]
          viewdirs: jnp.ndarray [B, 3]
          coarse: if true, uses coarse MLP
        Returns:
          raw_rgb: jnp.ndarray [B, 3 * (sh_deg + 1)**2 or 3 or 3 * sg_dim]
          raw_sigma: jnp.ndarray [B, 1]
        """
        points = points[None] #1,1w,3
        # points_enc = model_utils.posenc(
        #     points,
        #     self.min_deg_point,
        #     self.max_deg_point,
        #     self.legacy_posenc_order,
        # )
        
        points_enc=self.point_encoder(points) #nerfies 1,1w,51
        if self.num_fine_samples > 0 and not coarse:
            mlp = self.MLP_1
        else:
            mlp = self.MLP_0
        if self.use_viewdirs:
            assert viewdirs is not None
            viewdirs = viewdirs[None]
            # viewdirs_enc = model_utils.posenc(
            #     viewdirs,
            #     0,
            #     self.deg_view,
            #     self.legacy_posenc_order,
            # )
            viewdirs_enc=self.viewdir_encoder(viewdirs) # nerfies
            raw_rgb, raw_sigma = mlp(points_enc, viewdirs_enc)
        else:
            raw_rgb, raw_sigma = mlp(points_enc) # 1,1w,51  out 1,1w,48, 1,1w,1
        return raw_rgb[0], raw_sigma[0]

    def eval_points(self, points, viewdirs=None, coarse=False):
        """
        Evaluate at points, converting spherical harmonics rgb to
        rgb via viewdirs if applicable. Exists since jax does not allow
        size to depend on input.
        Args:
          points: jnp.ndarray [B, 3]
          viewdirs: jnp.ndarray [B, 3]
          coarse: if true, uses coarse MLP
        Returns:
          rgb: jnp.ndarray [B, 3]
          sigma: jnp.ndarray [B, 1]
        """
        raw_rgb, raw_sigma = self.eval_points_raw(points, viewdirs, coarse)
        if self.sh_deg >= 0:
            assert viewdirs is not None
            # (256, 64, 48) (256, 3)
            raw_rgb = sh.eval_sh(self.sh_deg, raw_rgb.reshape(
                *raw_rgb.shape[:-1],
                -1,
                (self.sh_deg + 1) ** 2), viewdirs[:, None])
        elif self.sg_dim > 0:
            assert viewdirs is not None
            sg_lambda = self.sg_lambda.value
            sg_mu_spher = self.sg_mu_spher.value
            sg_coeffs = raw_rgb.reshape(*raw_rgb.shape[:-1], -1, self.sg_dim)
            raw_rgb = sg.eval_sg(
                sg_lambda, sg_mu_spher, sg_coeffs, viewdirs[:, None])

        rgb = self.rgb_activation(raw_rgb)
        sigma = self.sigma_activation(raw_sigma)
        return rgb, sigma

    # def render_samples(self,
    #                  points):
        
    #     # Apply the deformation field to the samples.
    #     # if use_warp:
    #     #     metadata_channels = self.num_warp_features if metadata_encoded else 1
    #     #     warp_metadata = (
    #     #         metadata['time']
    #     #         if self.warp_metadata_encoder_type == 'time' else metadata['warp'])
    #     #     warp_metadata = jnp.broadcast_to(
    #     #         warp_metadata[:, jnp.newaxis, :],
    #     #     shape=(*points.shape[:2], metadata_channels))
    #     #     warp_out = self.warp_field(
    #     #         points,
    #     #         warp_metadata,
    #     #         warp_extra,
    #     #         use_warp_jacobian,
    #     #         metadata_encoded)
    #     #     points = warp_out['warped_points']
    #     points_embed=self.point_encoder(points)
    #     return points_embed
    
    
    def __call__(self, rng_0, rng_1, rays, randomized):
        """Nerf Model.

        Args:
          rng_0: jnp.ndarray, random number generator for coarse model sampling.
          rng_1: jnp.ndarray, random number generator for fine model sampling.
          rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
          randomized: bool, use randomized stratified sampling.

        Returns:
          ret: list, [(rgb_coarse, disp_coarse, acc_coarse), (rgb, disp, acc)]
        """
        # Stratified sampling along rays
        key, rng_0 = random.split(rng_0)
        z_vals, samples = model_utils.sample_along_rays(
            key,
            rays.origins,
            rays.directions,
            self.num_coarse_samples,
            self.near,
            self.far,
            randomized,
            self.lindisp,
        )
        
        # frequency band encoding
        samples=self.warp_field(samples)
        samples_enc=self.point_encoder(samples) #nerfies in 1024,64,3 out 1024,64,51

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_enc=self.viewdir_encoder(rays.viewdirs) # nerfies
            # viewdirs_enc_ori = model_utils.posenc(
            #     rays.viewdirs,
            #     0,
            #     self.deg_view,
            #     self.legacy_posenc_order,
            # )
            raw_rgb, raw_sigma = self.MLP_0(samples_enc, viewdirs_enc)
        else:
            raw_rgb, raw_sigma = self.MLP_0(samples_enc)# 1024,64,48  & 1024,64,1
        # Add noises to regularize the density predictions if needed
        key, rng_0 = random.split(rng_0)
        raw_sigma = model_utils.add_gaussian_noise(
            key,
            raw_sigma,
            self.noise_std,
            randomized,
        )

        if self.sh_deg >= 0:
            # (256, 64, 48) (256, 3)
            raw_rgb = sh.eval_sh(self.sh_deg, raw_rgb.reshape(
                *raw_rgb.shape[:-1],
                -1,
                (self.sh_deg + 1) ** 2), rays.viewdirs[:, None])
        elif self.sg_dim > 0:
            sg_lambda = self.sg_lambda.value
            sg_mu_spher = self.sg_mu_spher.value
            sg_coeffs = raw_rgb.reshape(*raw_rgb.shape[:-1], -1, self.sg_dim)
            raw_rgb = sg.eval_sg(
                sg_lambda, sg_mu_spher, sg_coeffs, rays.viewdirs[:, None])

        rgb = self.rgb_activation(raw_rgb)
        sigma = self.sigma_activation(raw_sigma)

        # Volumetric rendering.
        comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
            rgb, # 1024,64,3
            sigma, # 1024,64,1
            z_vals, #1024,64
            rays.directions, #1024,3
            white_bkgd=self.white_bkgd,
        )
        ret = [
            (comp_rgb, disp, acc),
        ]
        # Hierarchical sampling based on coarse predictions
        if self.num_fine_samples > 0:
            z_vals_mid = 0.5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
            key, rng_1 = random.split(rng_1)
            z_vals, samples = model_utils.sample_pdf(
                key,
                z_vals_mid,
                weights[Ellipsis, 1:-1],
                rays.origins,
                rays.directions,
                z_vals,
                self.num_fine_samples,
                randomized,
            )
            # samples_enc = model_utils.posenc(
            #     samples,
            #     self.min_deg_point,
            #     self.max_deg_point,
            #     self.legacy_posenc_order,
            # )
            samples_enc=self.point_encoder(samples) # nerfies

            if self.use_viewdirs:
                raw_rgb, raw_sigma = self.MLP_1(samples_enc, viewdirs_enc)
            else:
                raw_rgb, raw_sigma = self.MLP_1(samples_enc)
            key, rng_1 = random.split(rng_1)
            raw_sigma = model_utils.add_gaussian_noise(
                key,
                raw_sigma,
                self.noise_std,
                randomized,
            )
            if self.sh_deg >= 0:
                raw_rgb = sh.eval_sh(self.sh_deg, raw_rgb.reshape(
                    *raw_rgb.shape[:-1],
                    -1,
                    (self.sh_deg + 1) ** 2), rays.viewdirs[:, None])
            elif self.sg_dim > 0:
                sg_lambda = self.sg_lambda.value
                sg_mu_spher = self.sg_mu_spher.value
                sg_coeffs = raw_rgb.reshape(*raw_rgb.shape[:-1], -1, self.sg_dim)
                raw_rgb = sg.eval_sg(
                    sg_lambda, sg_mu_spher, sg_coeffs, rays.viewdirs[:, None])

            rgb = self.rgb_activation(raw_rgb)
            sigma = self.sigma_activation(raw_sigma)
            comp_rgb, disp, acc, unused_weights = model_utils.volumetric_rendering(
                rgb,
                sigma,
                z_vals,
                rays.directions,
                white_bkgd=self.white_bkgd,
            )
            ret.append((comp_rgb, disp, acc))
        return ret


def construct_nerf(key, args):
    """Construct a Neural Radiance Field.

    Args:
      key: jnp.ndarray. Random number generator.
      args: FLAGS class. Hyperparameters of nerf.

    Returns:
      model: nn.Model. Nerf model with parameters.
      state: flax.Module.state. Nerf model state for stateful parameters.
    """
    net_activation = getattr(nn, str(args.net_activation))
    rgb_activation = getattr(nn, str(args.rgb_activation))
    sigma_activation = getattr(nn, str(args.sigma_activation))

    # Assert that rgb_activation always produces outputs in [0, 1], and
    # sigma_activation always produce non-negative outputs.
    x = jnp.exp(jnp.linspace(-90, 90, 1024))
    x = jnp.concatenate([-x[::-1], x], 0)

    rgb = rgb_activation(x)
    if jnp.any(rgb < 0) or jnp.any(rgb > 1):
        raise NotImplementedError(
            "Choice of rgb_activation `{}` produces colors outside of [0, 1]".format(
                args.rgb_activation
            )
        )

    sigma = sigma_activation(x)
    if jnp.any(sigma < 0):
        raise NotImplementedError(
            "Choice of sigma_activation `{}` produces negative densities".format(
                args.sigma_activation
            )
        )
    num_rgb_channels = args.num_rgb_channels
    # TODO cleanup assert
    if args.sh_deg >= 0:
        assert not args.use_viewdirs and args.sg_dim == -1, (
                "You can only use up to one of: SH, SG or use_viewdirs.")
        num_rgb_channels *= (args.sh_deg + 1) ** 2
    elif args.sg_dim > 0:
        assert not args.use_viewdirs and args.sh_deg == -1, (
                "You can only use up to one of: SH, SG or use_viewdirs.")
        num_rgb_channels *= args.sg_dim

    model = NerfModel(
        min_deg_point=args.min_deg_point,
        max_deg_point=args.max_deg_point,
        deg_view=args.deg_view,
        num_coarse_samples=args.num_coarse_samples,
        num_fine_samples=args.num_fine_samples,
        use_viewdirs=args.use_viewdirs,
        sh_deg=args.sh_deg,
        sg_dim=args.sg_dim,
        near=args.near,
        far=args.far,
        noise_std=args.noise_std,
        white_bkgd=args.white_bkgd,
        net_depth=args.net_depth,
        net_width=args.net_width,
        net_depth_condition=args.net_depth_condition,
        net_width_condition=args.net_width_condition,
        skip_layer=args.skip_layer,
        num_rgb_channels=num_rgb_channels,
        num_sigma_channels=args.num_sigma_channels,
        lindisp=args.lindisp,
        net_activation=net_activation,
        rgb_activation=rgb_activation,
        sigma_activation=sigma_activation,
        legacy_posenc_order=args.legacy_posenc_order,
        num_nerf_point_freqs=args.num_nerf_point_freqs,
        num_nerf_viewdir_freqs=args.num_nerf_viewdir_freqs,
        use_warp=args.use_warp
    )
    init_rays_dict = {
      'origins': jnp.ones((args.batch_size, 3), jnp.float32),
      'directions': jnp.ones((args.batch_size, 3), jnp.float32),
      'metadata': {
          'warp': jnp.ones((args.batch_size, 1), jnp.uint32),
          'camera': jnp.ones((args.batch_size, 1), jnp.uint32),
          'appearance': jnp.ones((args.batch_size, 1), jnp.uint32),
          'time': jnp.ones((args.batch_size, 1), jnp.float32),
      }
    }
    warp_extra = {
        'alpha': 0.0,
        'time_alpha': 0.0,
    }
    key, key1, key2  = random.split(key,3)
    init_variables = model.init(
        key,
        method=model._quick_init,
    )
    init_variables = model.init({
      'params': key,
      'coarse': key1,
      'fine': key2
  },
                      init_rays_dict,
                      warp_extra=warp_extra)['params']
    return model, init_variables
