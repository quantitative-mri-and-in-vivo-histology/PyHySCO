"""Minimal JAX implementation of the *per‑slice* least‑squares
problem that EPIMRIDistortionCorrectionPush4dProper solves in its
`solve_lsq_slice` routine.

The core task is to reconstruct a single corrected image slice `rho`
that lives on *particles* (reconstruction grid) from several distorted
observations that live on *cells* (original EPI grid, one per phase‑
encoding direction).  Each observation k is linked to `rho` through a
matrix–free push‑forward operator T_k.  We solve

    argmin_rho  1/2 Σ_k ||T_k rho − y_k||²  +  λ_smooth ||∇rho||²

with a matrix‑free pre‑conditioned conjugate‑gradient (PCG).
Only the math kernels needed for that sub‑problem are implemented – no
field‑map handling, no Jacobians, no intensity regulariser, no proximal
term.  The code is ~200 lines and is ready to JIT.

Author: ChatGPT (OpenAI o3)
Date  : 2025‑06‑27
"""

from __future__ import annotations

from functools import partial
from typing import Sequence, Tuple, NamedTuple

import os

from scipy.special import jnp_zeros
from sympy.physics.vector import outer

# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_tracing=true"

# os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false --xla_gpu_autotune_level=3"
# os.environ["XLA_FLAGS"] = " --xla_disable_hlo_fusion=true"

import gc

import torch
import jax
import jax.numpy as jnp
from jax import lax
import jax.profiler
from jax import tree_util
from jax.profiler import annotate_function
from jax.experimental import sparse
from jax.scipy.sparse.linalg import cg
# from jax.scipy.optimize import minimizes
import os
import time
import optax
import jaxopt
# from triton.backends.nvidia.compiler import min_dot_size

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_check_tracer_leaks", True)

# jax.config.update("jax_disable_jit", True)

from EPI_MRI.LinearOperators import myAvg1D
from EPI_MRI.MultiPeDtiData import MultiPeDtiData
from EPI_MRI.InitializationMethods import InitializeCFMultiePeDtiDataResampled
from EPI_MRI.utils import save_data
import nibabel as nib
from typing import NamedTuple
from particle_in_cell_utils_autodiff import *
from jax.tree_util import register_pytree_node_class
from jaxopt import BFGS, LBFGS
from jax import value_and_grad
from jax.flatten_util import ravel_pytree

class Params(NamedTuple):
    bc: jnp.ndarray        # fieldmap shift: shape (3, Dz, Dy, Dx)
    theta: jnp.ndarray

tree_util.register_pytree_node(
    Params,
    lambda x: ((x.bc, x.theta), None),
    lambda _, children: Params(*children),
)



def pose_to_matrix(theta_row):
    t = theta_row[:3]
    rx, ry, rz = theta_row[3:]
    cx, sx = jnp.cos(rx), jnp.sin(rx)
    cy, sy = jnp.cos(ry), jnp.sin(ry)
    cz, sz = jnp.cos(rz), jnp.sin(rz)
    R = jnp.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])
    M = jnp.eye(4)
    M = M.at[:3, :3].set(R)
    M = M.at[:3, 3].set(t)
    return M


@register_pytree_node_class
class DistortionModel(NamedTuple):
    bc: jnp.ndarray
    thetas: jnp.ndarray
    phase_signs: jnp.ndarray
    pe_mats: jnp.ndarray

    def apply(self, xp_base, obs_index, volume_index):
        image_center = 0.5 * (jnp.array(data.omega[3::2]) + jnp.array(data.omega[2::2]))
        phase_sign = self.phase_signs[obs_index, volume_index]
        theta = self.thetas[obs_index, volume_index]
        motion_transform = pose_to_matrix(theta).astype(jnp.float32)
        ref_mat = self.pe_mats[0][0]
        roi_mat = self.pe_mats[obs_index, volume_index]
        T_mat_permuted = jnp.linalg.inv(roi_mat) @ ref_mat
        rot = T_mat_permuted[:3, :3]

        # xp_flat = xp_base.reshape(3, -1)  # shape (3, N)
        # ones = jnp.ones((1, xp_flat.shape[1]), dtype=jnp.float32)
        # xp_hom = jnp.vstack([xp_flat, ones])
        # xp_moved = motion_transform @ xp_hom
        # xp = xp_moved[:3].reshape(xp_base.shape)

        # directly use the bc from this distortion model
        pe_vec = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        xp = xp_base + phase_sign * (pe_vec[:, None, None, None] * self.bc)
        xp = rot @ (xp.reshape(3, -1) - image_center[:, None]) + image_center[:, None]

        # ones = jnp.ones((1, xp.shape[1]), dtype=jnp.float32)
        # xp_hom = jnp.vstack([xp, ones])  # shape (4, N)
        # xp_moved = motion_transform @ xp_hom
        # xp = xp_moved[:3, :]

        return xp

    def apply_backward(self, xp_rotated, obs_index, volume_index):
        image_center = 0.5 * (
                    jnp.array(data.omega[3::2]) + jnp.array(data.omega[2::2]))
        phase_sign = self.phase_signs[obs_index, volume_index]
        theta = self.thetas[obs_index, volume_index]
        ref_mat = self.pe_mats[0][0]
        roi_mat = self.pe_mats[obs_index, volume_index]
        T_mat_permuted = jnp.linalg.inv(roi_mat) @ ref_mat
        rot = T_mat_permuted[:3, :3]
        rot_inv = jnp.linalg.inv(rot)

        # undo the rotation
        xp_unrotated = rot_inv @ (xp_rotated.reshape(3, -1) - image_center[:,
                                                              None]) + image_center[
                                                                       :, None]

        # undo the fieldmap shift
        pe_vec = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        xp_base = xp_unrotated - phase_sign * (
                    pe_vec[:, None] * self.bc.reshape(3, -1))

        return xp_base.reshape(xp_rotated.shape)

    def tree_flatten(self):
        # Return dynamic children, static aux_data
        children = (self.bc, self.thetas, self.phase_signs, self.pe_mats)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class DwiOptimizer:
    def __init__(self, omega: jnp.ndarray, m_part: Tuple[int, int, int], m_cell: Tuple[int, int, int]):
        self.omega = omega
        self.m_part = m_part
        self.m_cell = m_cell
        self.n_cells = self.m_cell[0] * self.m_cell[1] * self.m_cell[2]
        self.n_particles = self.m_part[0] * self.m_part[1] * self.m_part[2]
        self.extent = omega[1::2] - omega[::2]
        self.part_sz = self.extent / jnp.array(m_part)
        self.cell_sz = self.extent / jnp.array(m_cell)
        self.inv_part_sz = 1.0 / self.part_sz
        self.xp_base = get_cell_centered_grid(omega, m_part, return_all=True)
        self.xp_base = self.xp_base.transpose(1, 0)
        self.xp_base = self.xp_base.reshape(3, *target_res[-3:])
        self.xp_base = self.xp_base.reshape(*self.xp_base.shape[0:3], -1)
        self.lambda_smooth = 0.2
        self.stencil_build_fn = make_stencil_builder(self.omega, self.m_cell, self.m_part)

        # self.stencil_build_fn = make_diff_stencil_builder(self.omega, self.m_cell, 1)
        self.part_lin = (
                (jnp.arange(self.m_part[0])[:, None, None] * self.m_part[1]
                 + jnp.arange(self.m_part[1])[None, :, None]) * self.m_part[2]
                + jnp.arange(self.m_part[2])[None, None, :]).reshape(-1)
        # ------------------------------------------------------------


    @partial(jax.jit, static_argnums=(0))
    def solve(self, vol_obs: jnp.ndarray,
              distortion_model):
        """
        Runs ATAx-based PIC-LSQ for each volume (via lax.map).
        """
        n_vols = vol_obs.shape[1]
        n_obs = vol_obs.shape[0]
        M_inv = lambda r: r


        @jax.checkpoint
        @partial(jax.jit, static_argnums=())  # Nothing static
        def solve_one(vol_index, distortion_model):
            with jax.profiler.TraceAnnotation("solve_one"):
                xp_base = self.xp_base + 0.0 * distortion_model.bc
                # rho_init = jnp.zeros(
                #     self.m_part[0] * self.m_part[1] * self.m_part[2])
                rho_init = jnp.zeros(
                    self.m_part[0] * self.m_part[1] * self.m_part[2],
                    dtype=jnp.float32)

                y = vol_obs[:, vol_index].reshape(vol_obs.shape[0], -1)
                rhs = self.ATy(y, vol_index, xp_base, distortion_model)
                # M_inv = self._build_precond(distortion_model, vol_index, n_obs)
                # ATAx_fn = partial(self.ATAx,
                #                   vol_index=vol_index,
                #                   n_obs=n_obs,
                #                   xp_base=xp_base,
                #                   lambda_smooth=self.lambda_smooth,
                #                   distortion_model=distortion_model)

                def ATAx_fn(v):
                    return self.ATAx(v, vol_index, n_obs, xp_base,
                                     self.lambda_smooth, distortion_model)

                rho_est, _ = jax.scipy.sparse.linalg.cg(ATAx_fn, rhs, M=M_inv, x0=rho_init,
                                                        tol=1e-3, maxiter=10)
                rho_est = rho_est.astype(jnp.float32)

                # push_fn = make_pushforward_fn(self.stencil_build_deriv_fn)

                # True data residual: ∑ₖ ‖Tₖ rho − yₖ‖²
                def forward_error(k, err_sum):
                    xp = distortion_model.apply(xp_base, k, vol_index)
                    idx, w = self.stencil_build_fn(xp)
                    # y_pred = jnp.sum(rho_est[idx] * w, axis=1)
                    # # y_pred = push_fn(rho_est, xp)
                    # err = jnp.sum((y_pred - y[k]) ** 2)

                    y_k_ref = pic_adjoint(y[k], idx, w)  # resample y_k into reference frame
                    err = jnp.sum((rho_est - y_k_ref) ** 2)

                    return err_sum + err

                loss = lax.fori_loop(0, n_obs, forward_error, 0.0)

                # loss = jnp.sum((ATAx_fn(rho_est) - rhs) ** 2)

                loss = jnp.sum((rho_est - rhs) ** 2)
                loss = loss.astype(jnp.float32)

                return rho_est, loss

        # solve_all = lambda i: solve_one(i, distortion_model)
        # rho_all, loss_all = lax.map(solve_all, jnp.arange(n_vols), batch_size=1)

        rho_all, loss_all = lax.map(lambda i: solve_one(i, distortion_model),
                                    jnp.arange(n_vols), batch_size=1)

        # rho_all, loss_all = lax.map(solve_one, jnp.arange(n_vols), batch_size=1)
        # rho_all, loss_all = lax.map(solve_one, jnp.arange(5),
        #                             batch_size=5)
        total_loss = jnp.mean(loss_all)
        rho_all = rho_all.astype(jnp.float32)
        total_loss = total_loss.astype(jnp.float32)

        return rho_all, total_loss

    @partial(jax.jit, static_argnums=(0))
    def ATy(self,
            y: jnp.ndarray,  # shape (n_obs, n_cells)
            vol_index: int,
            xp_base,
            distortion_model) -> jnp.ndarray:
        """Compute RHS rhs = (1/n_obs) Σ_k T_kᵀ y_k  in JAX-friendly form."""

        n_obs = y.shape[0]  # traced as a constant inside jit

        def body(k, acc):
            # --- 1. particle coords for this observation ------------------------
            xp = distortion_model.apply(xp_base, k, vol_index)  # (3, Np)

            # --- 2. build 27-point PIC stencil ----------------------------------
            idx, w = self.stencil_build_fn(xp)

            # --- 3. apply adjoint ----------------------------------------------
            acc = acc + pic_adjoint(y[k], idx, w)
            return acc

        rhs = lax.fori_loop(0, n_obs, body,
                            jnp.zeros(self.n_particles, dtype=y.dtype))
        return rhs / n_obs

    # ------------------------------------------------------------------
    #  v-mapped ATAx  (each vmap lane handles ONE observation)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0, 3, 5))
    def ATAx(self, x, vol_index, n_obs, xp_base, lambda_smooth, distortion_model):
        """
        Computes       AᵀA x = (1/n_obs) Σ_k  T_kᵀ T_k x  + λ Δx
        with one vmap lane per observation.  No large per-obs buffers are
        kept; everything is built and discarded inside the lane.
        """

        # --------------------------------------------------------------
        # inner kernel for ONE observation k
        # --------------------------------------------------------------
        def one_obs(k, x_vec):
            """
            k      : scalar int32  (observation index)
            x_vec  : (N_particles,)
            return : (N_particles,)   result of T_kᵀ T_k x_vec
            """
            # 1) build PIC stencil for this observation
            xp = distortion_model.apply(xp_base, k, vol_index)  # (3,N)

            # xp_vox = jnp.floor(xp).astype(int)
            # z, y, x = xp_vox
            # in_patch = (
            #         (z >= 14) & (z < 20) &
            #         (y >= 30) & (y < 35) &
            #         (x >= 30) & (x < 35)
            # )
            # jax.debug.print("Particles in patch: {}", jnp.sum(in_patch))

            idx, w = self.stencil_build_fn(xp)  # (N,S) each

            # 2) forward ρ → y   (particles ➜ cells)
            val_flat = (x_vec[:, None] * w).reshape(-1)  # (N*S,)
            idx_flat = idx.reshape(-1)
            y_cells = jnp.zeros(self.n_cells, x.dtype).at[idx_flat].add(val_flat)

            # 3) adjoint  y → ρ  (cells ➜ particles)
            contrib = y_cells[idx] * w  # (N,S)
            yt = contrib.sum(axis=1)  # (N,)

            return yt.astype(x_vec.dtype)

        # vmap over k:  in_axes=(0, None)  → k varies, x is broadcast
        TtT_all = jax.vmap(one_obs, in_axes=(0, None))(  # (n_obs, N)
            jnp.arange(n_obs, dtype=jnp.int32), x)
        TtT = TtT_all.mean(axis=0)  # (N,)

        # 4) smoothness term
        if lambda_smooth > 0.0:
            TtT += lambda_smooth * tikhonov_like_3d_full(x, self.m_part)

        return TtT

    # @partial(jax.jit, static_argnums=(0, 1, 3))
    def _build_precond(self, xp_base, distortion_model, vol_index: int, n_obs):
        """
        Return a function  M(x) = D⁻¹·x
        where D = diag( (1/n_obs) Σ_k T_kᵀ T_k + λ_s Δ ) for a given volume.
        """
        lambda_s = self.lambda_smooth

        def obs_contrib(k, acc):
            xp = distortion_model.apply(xp_base, k,
                                        vol_index)  # ← volume-aware
            idx, w = self.stencil_build_fn(xp)
            acc += jnp.sum(w ** 2, axis=1)
            return acc

        diag = lax.fori_loop(
            0, n_obs, obs_contrib,
            jnp.zeros(self.n_particles, dtype=jnp.float32)
        ) / n_obs

        diag += lambda_s * 6.0
        inv_diag = 1.0 / jnp.where(diag > 1e-6, diag, 1e-6)
        return lambda x: inv_diag * x


def save_jax_data(data, filepath, affine=None):
    """
    Save data to the given filepath.

    Parameters
    ----------
    data : torch.Tensor (size any)
        Data to save.
    filepath : str
        Path where to save data.
    """
    if affine is None:
        affine = np.eye(4)
    if filepath is not None:
        save_img = nib.Nifti1Image(data, np.asarray(affine))
        nib.save(save_img, filepath)


@jax.jit
def pic_adjoint(y: jnp.ndarray,
                indices: jnp.ndarray,
                weights: jnp.ndarray) -> jnp.ndarray:
    """Compute rho = Tᵗ y from cell-space y."""
    contrib = y[indices] * weights       # shape: (N_particles, K)
    return jnp.sum(contrib, axis=1)      # shape: (N_particles,)



################################################################################
# 4.  Diagonal pre‑conditioner                                                  #
################################################################################

def diag_precond(indices: jnp.ndarray, weights: jnp.ndarray, n_particles: int) -> jnp.ndarray:
    """Return   diag(Tᵀ T)   to be used as a Jacobi PC."""
    w2 = jnp.sum(weights ** 2, axis=[0,2])
    return jnp.maximum(w2, 1e-6)             # avoid divide‑by‑0


################################################################################
# 6.  Public API:  solve_lsq_slice                                              #
################################################################################

# @partial(jax.jit, static_argnums=(1))
def tikhonov_like_3d_full(r, spatial_size):

    D, H, W = spatial_size
    U = r.reshape(D, H, W)

    # Forward differences (roll-based)
    dz = jnp.roll(U, -1, axis=0) - U
    dy = jnp.roll(U, -1, axis=1) - U
    dx = jnp.roll(U, -1, axis=2) - U

    # Zero out forward diffs at boundaries
    dz = dz.at[-1, :, :].set(0.0)
    dy = dy.at[:, -1, :].set(0.0)
    dx = dx.at[:, :, -1].set(0.0)

    # Adjoint (backward diff of forward diff)
    dzt = -(dz - jnp.roll(dz, 1, axis=0))
    dyt = -(dy - jnp.roll(dy, 1, axis=1))
    dxt = -(dx - jnp.roll(dx, 1, axis=2))

    # Fix first slice/row/col (they have no backward neighbour)
    dzt = dzt.at[0, :, :].set(0.0)
    dyt = dyt.at[:, 0, :].set(0.0)
    dxt = dxt.at[:, :, 0].set(0.0)

    return (dzt + dyt + dxt).reshape(-1)


def laplacian_conv(x, spatial):
    D,H,W = spatial
    K = jnp.array([[[0,0,0],
                    [0,1,0],
                    [0,0,0]],
                   [[0,1,0],
                    [1,-6,1],
                    [0,1,0]],
                   [[0,0,0],
                    [0,1,0],
                    [0,0,0]]], x.dtype)
    return lax.conv_general_dilated(
            x.reshape(1,1,D,H,W),
            K.reshape(1,1,3,3,3),
            window_strides=(1,1,1),
            padding="SAME")[0,0].reshape(-1)



def laplacian_3d(v):
    """Discrete 3-D Laplacian with Neumann (zero-grad) boundaries.

    v: (..., D, H, W)  or  (..., H, W)   (it works for both)
    returns same shape
    """
    # forward differences
    dz = jnp.roll(v, -1, axis=-3) - v if v.ndim >= 3 else 0.0
    dy = jnp.roll(v, -1, axis=-2) - v
    dx = jnp.roll(v, -1, axis=-1) - v

    # zero at far boundary
    if v.ndim >= 3:
        dz = dz.at[..., -1, :, :].set(0.0)
    dy = dy.at[..., -1, :].set(0.0)
    dx = dx.at[..., :, -1].set(0.0)

    # backward diff of forward diff
    dzt = -(dz - jnp.roll(dz, 1, axis=-3)) if v.ndim >= 3 else 0.0
    dyt = -(dy - jnp.roll(dy, 1, axis=-2))
    dxt = -(dx - jnp.roll(dx, 1, axis=-1))

    if v.ndim >= 3:
        dzt = dzt.at[..., 0, :, :].set(0.0)
    dyt = dyt.at[..., 0, :].set(0.0)
    dxt = dxt.at[..., :, 0].set(0.0)

    return dzt + dyt + dxt


def smooth_loss_fn(x):
    lap = laplacian_3d(x)
    return jnp.sum(lap ** 2)




def loss_fn(params: Params,
            xp_base: jnp.ndarray,
            dwi_images: jnp.ndarray,
            omega: jnp.ndarray,
            m_part: Tuple[int, int, int],
            m_cell: Tuple[int, int, int],
            phase_signs: jnp.ndarray,
            pe_mats: jnp.ndarray,
            weight_bc: float = 1e-3,
            weight_motion: float = 1e-4):

    # 1) rebuild distortion model with current (bc, theta)
    distortion_model = DistortionModel(params.bc, params.theta, phase_signs, pe_mats)

    # 2) run the PIC–LSQ solver (differentiable through CG)
    optim = DwiOptimizer(omega, m_part=m_part, m_cell=m_cell)
    recon, data_term = optim.solve(dwi_images, distortion_model)  # data_term already ∑k‖Tρ−y‖²


    # 3) field-map smoothness   (∥∇bc∥² in voxel space)
    bc_smooth = jnp.sum(laplacian_3d(params.bc) ** 2)

    # 4) motion quadratic penalty
    motion_penalty = jnp.mean(params.theta ** 2)

    # 5) total loss
    total = data_term + weight_bc * bc_smooth + weight_motion * motion_penalty
    return total, (data_term, bc_smooth, motion_penalty, recon)




if __name__ == "__main__":
    import numpy as np

    # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # logdir = f"/home/laurin/workspace/PyHySCO/data/results/debug/tensor_logs/{timestamp}"
    # os.makedirs(logdir, exist_ok=True)
    # # jax.profiler.trace_memory()
    # jax.profiler.start_trace(logdir)
    # # start_time = time.time()

    image_config_file = "/home/laurin/workspace/PyHySCO/data/raw/highres/image_config.json"
    device =  'cpu'
    pair_idx = [0,1,2,3]
    # pair_idx = 0
    data = MultiPeDtiData(image_config_file, device=device, dtype=torch.float32, pair_idx=pair_idx)

    target_res = [*data.m[:-2], 128, 128]
    target_res = torch.tensor(target_res, dtype=torch.int32, device=device)
    initialization = InitializeCFMultiePeDtiDataResampled()
    B0 = initialization.eval(data, target_res, blur_result=True)


    target_res = jnp.array(target_res.numpy())
    m_distorted = jnp.array(data.m.numpy())
    omega = data.omega.numpy()


    avg_op = myAvg1D(data.omega[2:], target_res[1:], device=device, dtype=torch.float32)
    # B0_res = B0.reshape(-1,1)
    bc = avg_op.mat_mul(B0).reshape(-1,1)

    v_pe = torch.tensor([0.0, 0.0, 1.0], device=device,
                             dtype=torch.float32)
    bc_3d = (bc * v_pe.view(1, -1)).T  # shift vector in original space
    bc_3d = bc_3d.reshape(3, *target_res[1:])
    bc_3d = jnp.array(bc_3d)

    xp_base = get_cell_centered_grid(omega[2:], target_res[1:], return_all=True)
    xp_base = xp_base.transpose(1, 0)
    xp_base = xp_base.reshape(3, *target_res[-3:])


    vols = []
    phase_signs = []
    phase_encoding_mats = []
    for pair_index, pair in enumerate(data.image_pairs):
        pe, rpe = pair
        pe_image_data = jnp.array(pe.data.numpy())
        rpe_image_data = jnp.array(rpe.data.numpy())
        vols.append(pe_image_data)
        vols.append(rpe_image_data)

        phase_signs.append(jnp.repeat(pe.phase_sign, pe_image_data.shape[0]))
        phase_signs.append(jnp.repeat(rpe.phase_sign, rpe_image_data.shape[0]))

        # pe_mat = jnp.array(pe.affine)
        pe_mat = jnp.array(data.mats[pair_index])
        pe_mats = pe_mat[None,:].repeat(pe_image_data.shape[0], axis=0)
        phase_encoding_mats.append(pe_mats)

        rpe_mat = jnp.array(data.mats[pair_index])
        rpe_mats = rpe_mat[None, :].repeat(rpe_image_data.shape[0], axis=0)
        phase_encoding_mats.append(rpe_mats)

    phase_signs = jnp.stack(phase_signs)
    phase_encoding_mats = jnp.stack(phase_encoding_mats)

    dwi_images = jnp.stack(vols)
    n_obs, n_vol, D, H, W = dwi_images.shape
    thetas = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1).repeat(dwi_images.shape[0]*dwi_images.shape[1],
                                                                          axis=0).reshape(*dwi_images.shape[:2], 6)

    target_res_tuple = (
    target_res[1].item(), target_res[2].item(), target_res[3].item())
    m_distorted_tuple = (
    m_distorted[1].item(), m_distorted[2].item(), m_distorted[3].item())


    omega_3d = jnp.array(omega[2:], dtype=jnp.float32)

    params = Params(
        bc=bc_3d,  # your initial fieldmap
        theta=thetas  # zeros of shape (N_pairs, 6)
    )

    # ---- 1. Flatten the Params ----
    params_flat, unravel_fn = ravel_pytree(params)


    # ---- 2. Define flat loss fn for BFGS ----
    def loss_fn_flat(x_flat):
        params_unflat = unravel_fn(x_flat)
        total_loss, aux = loss_fn(
            params_unflat,
            xp_base, dwi_images,
            omega_3d,
            target_res_tuple, m_distorted_tuple,
            phase_signs, phase_encoding_mats,
            weight_bc=100.0,
            weight_motion=1e10,
        )
        return jnp.atleast_1d(total_loss)[0]


    loss_fn_flat_grad = value_and_grad(loss_fn_flat)


    # ---- 3. Optional callback to monitor progress ----
    def callback(xk, state):
        print(f"[BFGS] iter {state.iter_num:2d} | loss = {state.value:.4e}")
        return False  # return True to stop early


    # ---- 4. Run BFGS ----
    solver = LBFGS(fun=loss_fn_flat, value_and_grad=False, maxiter=30, verbose=True)
    result = solver.run(params_flat)
    params_opt = unravel_fn(result.params)
    final_loss = result.state.value
    print("Final loss:", final_loss)



    # bc_schedule = optax.exponential_decay(
    #     init_value=1e-2, transition_steps=50, decay_rate=0.95, staircase=True)
    # theta_schedule = optax.constant_schedule(1e-4)
    #
    # opt = optax.multi_transform(
    #     {
    #         'bc': optax.adam(bc_schedule),
    #         'theta': optax.adam(theta_schedule),
    #     },
    #     param_labels={'bc': 'bc', 'theta': 'theta'}
    # )
    #
    #
    #
    # # opt.init(params)
    # params_dict = {'bc': params.bc, 'theta': params.theta}
    # opt_state = opt.init(params_dict)
    # num_steps = 30
    # lambda_bc = 100
    # lambda_motion = 1e10
    # loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    #
    # for step in range(num_steps):
    #     (loss_val, (data_term, smooth_term, motion_term, recon)), grads = \
    #         loss_grad_fn(params, xp_base, dwi_images,
    #                      omega_3d,  # same as `omega`
    #                      target_res_tuple, m_distorted_tuple,
    #                      phase_signs, phase_encoding_mats,
    #                      weight_bc=lambda_bc,
    #                      weight_motion=lambda_motion)
    #
    #     grads_dict = {'bc': grads.bc, 'theta': grads.theta}
    #     updates, opt_state = opt.update(grads_dict, opt_state, params_dict)
    #
    #     # Apply update:
    #     params_dict = optax.apply_updates(params_dict, updates)
    #
    #     # Optional: rewrap into Params
    #     params = Params(**params_dict)
    #
    #
    #     print(f"step {step:3d} | L={loss_val:9.3e} | "
    #           f"D={data_term:9.3e}  S={smooth_term:9.3e}  M={motion_term:9.3e}")
    #
    #
    #     save_jax_data(recon.reshape(*target_res).transpose(3, 2, 1, 0),
    #                       f"/home/laurin/workspace/PyHySCO/data/results/debug/recon_moco_first_{step}.nii.gz", data.mats[0])
    #     save_jax_data(params.bc.reshape(3, *target_res[1:]).transpose(3, 2, 1, 0),
    #                   f"/home/laurin/workspace/PyHySCO/data/results/debug/bc_{step}.nii.gz")
    #     save_jax_data(grads.bc.reshape(3, *target_res[1:]).transpose(3, 2, 1, 0),
    #                   f"/home/laurin/workspace/PyHySCO/data/results/debug/grad_bc_{step}.nii.gz")


    # # Stop trace recording
    # jax.profiler.stop_trace()
    #
    # jax.profiler.save_device_memory_profile(os.path.join(logdir, "memory.prof"))