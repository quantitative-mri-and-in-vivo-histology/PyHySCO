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
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_memory_statistics"

import gc

import torch
import jax
import jax.numpy as jnp
from jax import lax
import jax.profiler
from jax.profiler import annotate_function
from jax.experimental import sparse
from jax.scipy.sparse.linalg import cg
# from jax.scipy.optimize import minimizes
import os
import time
import optax
import jaxopt
# from triton.backends.nvidia.compiler import min_dot_size

# jax.config.update("jax_disable_jit", True)

from EPI_MRI.LinearOperators import myAvg1D
from EPI_MRI.MultiPeDtiData import MultiPeDtiData
from EPI_MRI.InitializationMethods import InitializeCFMultiePeDtiDataResampled
from EPI_MRI.utils import save_data
import nibabel as nib
from typing import NamedTuple
from particle_in_cell_utils import *
from particle_in_cell_utils import _b_kernel_integrated
from scripts.lsq_jax_motion_per_pe import batch_solve


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


class DistortionModel:
    def __init__(self,
                 bc=None,
                 thetas=None,
                 phase_signs=None,
                 pe_mats=None):
        self.bc = bc.reshape(3, -1)
        self.thetas = thetas
        self.phase_signs = phase_signs
        self.pe_mats = pe_mats

    def update(self, params):
        self.bc = params.bc
        self.motion_transforms = params.motion_transforms

    @partial(jax.jit, static_argnums=(0))
    def apply(self, xp_base, obs_index, volume_index):

        image_center = 0.5 * (
                torch.tensor(data.omega[3::2]) + torch.tensor(
            data.omega[2::2]))  # (x_c, y_c, z_c)
        image_center = jnp.array(image_center.numpy())
        image_center_res = image_center.reshape(3,1,1,1)


        ref_mat = jnp.array(self.pe_mats[0][0])
        # bc = self.bc
        theta = self.thetas[obs_index, volume_index]
        motion_transform = pose_to_matrix(theta)

        T_motion_k_pe = motion_transform  # canonical-mm → moved canonical-mm

        roi_mat = jnp.array(self.pe_mats[obs_index, volume_index])
        T_mat_permuted = jnp.linalg.inv(roi_mat) @ ref_mat
        T_voxel_to_mm = jnp.eye(4) * (1 / jnp.array(
            [data.omega[3], data.omega[5], data.omega[7], 1]))
        T_mm_to_voxel = jnp.eye(4) * (jnp.array(
            [data.omega[3], data.omega[5], data.omega[7], 1]))
        T_fixed = T_voxel_to_mm @ T_mat_permuted @ T_mm_to_voxel
        rot_mat_permuted = T_mat_permuted[:3, :3]

        # xp_pe = xp_base.reshape(3, -1)
        # xp_lin = xp_pe + self.phase_signs[obs_index, volume_index] * self.bc
        # xp_lin = rot_mat_permuted @ (xp_lin - image_center.reshape(-1,
        #                                                            1)) + image_center.reshape(
        #     -1, 1)

        xp_lin = xp_base.reshape(3, -1) + self.phase_signs[obs_index, volume_index] * self.bc
        xp_lin = rot_mat_permuted @ (
                    xp_lin - image_center[:, None]) + image_center[:, None]

        return xp_lin



class FrameParams(NamedTuple):
    M: jnp.ndarray
    pe_sign: float
    eddy_coeff: jnp.ndarray

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
        self.lambda_smooth = 0.1
        self.stencil_build_fn = make_stencil_builder(self.omega, self.m_cell, self.m_part)
        self.part_lin = (
                (jnp.arange(self.m_part[0])[:, None, None] * self.m_part[1]
                 + jnp.arange(self.m_part[1])[None, :, None]) * self.m_part[2]
                + jnp.arange(self.m_part[2])[None, None, :]).reshape(-1)
        # ------------------------------------------------------------


    @partial(jax.jit, static_argnums=(0,2))
    def solve(self, vol_obs: jnp.ndarray,
              distortion_model):
        """
        Runs ATAx-based PIC-LSQ for each volume (via lax.map).
        """
        n_vols = vol_obs.shape[1]
        n_obs = vol_obs.shape[0]
        M_inv = lambda r: r

        def solve_one(vol_index):
            with jax.profiler.TraceAnnotation("solve_one"):
                rho_init = jnp.zeros(
                    self.m_part[0] * self.m_part[1] * self.m_part[2])

                y = vol_obs[:, vol_index].reshape(vol_obs.shape[0], -1)
                rhs = self.ATy(y, vol_index, distortion_model)
                ATAx_fn = partial(self.ATAx,
                                  vol_index=vol_index,
                                  n_obs=n_obs,
                                  lambda_smooth=self.lambda_smooth,
                                  distortion_model=distortion_model)
                rho_est, _ = jax.scipy.sparse.linalg.cg(ATAx_fn, rhs, M=M_inv, x0=rho_init,
                                                        tol=1e-5, maxiter=10)
                loss = jnp.sum((ATAx_fn(rho_est) - rhs) ** 2)
                return rho_est, loss

        rho_all, loss_all = lax.map(solve_one, jnp.arange(n_vols), batch_size=1)
        # rho_all, loss_all = lax.map(solve_one, jnp.arange(5),
        #                             batch_size=5)
        total_loss = jnp.sum(loss_all)
        return rho_all, total_loss

    @partial(jax.jit, static_argnums=(0, 3))
    def ATy(self,
            y: jnp.ndarray,  # shape (n_obs, n_cells)
            vol_index: int,
            distortion_model) -> jnp.ndarray:
        """Compute RHS rhs = (1/n_obs) Σ_k T_kᵀ y_k  in JAX-friendly form."""

        n_obs = y.shape[0]  # traced as a constant inside jit

        def body(k, acc):
            # --- 1. particle coords for this observation ------------------------
            xp = distortion_model.apply(self.xp_base, k, vol_index)  # (3, Np)

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
    @partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def ATAx(self, x, vol_index, n_obs, lambda_smooth, distortion_model):
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
            xp = distortion_model.apply(self.xp_base, k, vol_index)  # (3,N)
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
        save_img = nib.Nifti1Image(data, affine)
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


class Params(NamedTuple):
    bc: jnp.ndarray        # fieldmap shift: shape (3, Dz, Dy, Dx)
    theta: jnp.ndarray


if __name__ == "__main__":
    import numpy as np

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"/home/laurin/workspace/PyHySCO/data/results/debug/tensor_logs/{timestamp}"
    # os.makedirs(logdir, exist_ok=True)
    # jax.profiler.trace_memory()
    # jax.profiler.start_trace(logdir)
    # start_time = time.time()

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
    # thetas = jnp.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]).reshape(1, -1).repeat(dwi_images.shape[0]*dwi_images.shape[1],
    #                                                                       axis=0).reshape(*dwi_images.shape[:2], 6)
    thetas = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1).repeat(dwi_images.shape[0]*dwi_images.shape[1],
                                                                          axis=0).reshape(*dwi_images.shape[:2], 6)
    # thetas = thetas.transpose(1,0,2)

    distortion_model = DistortionModel(bc_3d, thetas, phase_signs, phase_encoding_mats)


    target_res_tuple = (
    target_res[1].item(), target_res[2].item(), target_res[3].item())
    m_distorted_tuple = (
    m_distorted[1].item(), m_distorted[2].item(), m_distorted[3].item())


    omega_3d = jnp.array(omega[2:], dtype=jnp.float32)

    # omega_3d_tuple = (
    # omega_3d[2].item(), omega_3d[3].item(), omega_3d[4].item(),
    # omega_3d[5].item(), omega_3d[6].item(), omega_3d[7].item())

    params = Params(
        bc=bc_3d,  # your initial fieldmap
        theta=thetas  # zeros of shape (N_pairs, 6)
    )

    optimizer = DwiOptimizer(omega_3d, m_part=target_res_tuple, m_cell=m_distorted_tuple)

    start_time = time.time()

    recon, data_loss = optimizer.solve(dwi_images, distortion_model)

    # Compute data term
    data_loss = jnp.mean(data_loss)

    end = time.time()
    print(f"Took {end - start_time:.4f} seconds")

    save_jax_data(recon.reshape(*target_res).transpose(3, 2, 1, 0),
                  f"/home/laurin/workspace/PyHySCO/data/results/debug/recon.nii.gz")

    # Wait briefly to ensure all ops complete
    # jax.block_until_ready(recon)
    #
    # # Stop trace recording
    # jax.profiler.stop_trace()
    #
    # jax.profiler.save_device_memory_profile(os.path.join(logdir, "memory.prof"))