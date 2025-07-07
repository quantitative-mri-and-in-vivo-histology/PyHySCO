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
from scripts.lsq_jax_motion_per_pe import batch_solve


# @partial(jax.jit, static_argnums=0)
# def build_cell_lists(xp,              # (3, N_particles) particle coords
#                      m_cell):         # (Dz, Dy, Dx)
#     """Return (cell_particles, part_cells) for a whole volume."""
#     Nz, Ny, Nx = m_cell
#     N_cells = Nz * Ny * Nx
#
#     # cell index for every particle
#     cz = (xp[0] // cell_sz[0]).astype(jnp.int32)
#     cy = (xp[1] // cell_sz[1]).astype(jnp.int32)
#     cx = (xp[2] // cell_sz[2]).astype(jnp.int32)
#     lin = (cz * Ny + cy) * Nx + cx            # (N_particles,)
#
#     # part_cells keeps (cz,cy,cx) for each particle
#     part_cells = jnp.stack([cz, cy, cx], axis=1)
#
#     # build cell_particles via segment-scatter
#     order = jnp.argsort(lin)                  # stable
#     lin_sorted = lin[order]
#     pid_sorted = order
#
#     # slots 0..MAX_PER_CELL-1 inside each cell
#     slot = jnp.cumsum(jnp.ones_like(lin_sorted)) - 1
#     slot = slot - jax.ops.segment_sum(slot, lin_sorted, N_cells)
#
#     # initialise with −1
#     cell_particles = -jnp.ones((N_cells, MAX_PER_CELL), dtype=jnp.int32)
#     cell_particles = cell_particles.at[lin_sorted, slot].set(pid_sorted)
#
#     return cell_particles, part_cells
#
#
#
# # ---------------------------------------------------------
# # Map: cell index -> list of particles inside each cell
# # ---------------------------------------------------------
# def particles_per_cell(cell_idx, n_cells, max_per_cell=128):
#     """Returns a (n_cells, max_per_cell) matrix with particle IDs."""
#     N = cell_idx.shape[0]
#     counts = jnp.zeros(n_cells, dtype=jnp.int32)
#     cell_particles = -jnp.ones((n_cells, max_per_cell), dtype=jnp.int32)
#
#     def body_fun(i, state):
#         counts, table = state
#         c = cell_idx[i]
#         idx = counts[c]
#         table = table.at[c, idx].set(i)
#         counts = counts.at[c].add(1)
#         return counts, table
#
#     _, table = jax.lax.fori_loop(0, N, body_fun, (counts, cell_particles))
#     return table  # shape (n_cells, max_per_cell)
#
#
# # ----------------------------------------------------------------
# # Get neighbor particle indices of particle p_idx using cell list
# # ----------------------------------------------------------------
# def neighbours_of(p_idx, cell_particles, part_cells, m_cell, pwidth=1):
#     """Returns particle indices in neighboring cells (flattened)."""
#     cell = part_cells[p_idx]
#
#     # Decode linear index back to 3D cell coords
#     cz = cell // (m_cell[1] * m_cell[2])
#     cy = (cell % (m_cell[1] * m_cell[2])) // m_cell[2]
#     cx = cell % m_cell[2]
#
#     offsets = jnp.array([
#         (dz, dy, dx)
#         for dz in range(-pwidth, pwidth + 1)
#         for dy in range(-pwidth, pwidth + 1)
#         for dx in range(-pwidth, pwidth + 1)
#     ])  # shape (M, 3)
#
# MAX_NBR = 27 * offsets.shape[0]  # e.g. 27·max_per_cell
#
# def neighbours_of(pid,
#                   cell_particles,
#                   part_cells,
#                   offsets,
#                   m_cell):
#     """Return (nbr_ids, mask) both fixed‒shape (MAX_NBR,)."""
#
#     cz, cy, cx = part_cells[pid]  # that particle’s cell
#
#     def get_neighbors(offset):
#         dz, dy, dx = offset  # each a scalar
#         nz, ny, nx = cz + dz, cy + dy, cx + dx
#         valid_cell = (0 <= nz) & (nz < m_cell[0]) & \
#                      (0 <= ny) & (ny < m_cell[1]) & \
#                      (0 <= nx) & (nx < m_cell[2])
#         ncell = (nz * m_cell[1] + ny) * m_cell[2] + nx
#         # pick all particles in that cell or −1 if cell is invalid
#         return jnp.where(valid_cell, cell_particles[ncell], -1)
#
#     # (n_offsets, max_per_cell)
#     nbr_block = jax.vmap(get_neighbors)(offsets)
#
#     # collapse to flat (MAX_NBR,)
#     nbr_ids = nbr_block.reshape(-1)
#
#     # build a concrete boolean mask
#     mask = nbr_ids >= 0
#
#     # keep sentinel −1 in nbr_ids; NEVER slice with the mask
#     nbr_ids = jnp.where(mask, nbr_ids, 0)  # 0 won’t be used when masked
#
#     return nbr_ids, mask  # both length MAX_NBR


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

    @partial(jax.jit, static_argnums=(0,3))
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

    # --------------------------------------------
    # Example of ATAx using cell list neighbor ops
    # --------------------------------------------
    @partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def ATAx(self, x, vol_index, n_obs, lambda_smooth, distortion_model):
        """Simplified ATAx using neighbor list logic."""

        def single_TtT(k):
            xp = distortion_model.apply(self.xp_base, k, vol_index)  # (3, N)
            cell_idx = build_cell_list(xp, self.m_cell, self.omega)
            cell_particles = particles_per_cell(cell_idx, self.n_cells)

            def per_particle(pid):
                nbr_ids, mask = neighbours_of(pid,
                                              cell_particles,
                                              part_cells,
                                              offsets,
                                              m_cell)

                # gather particle values once
                rho_nbr = x[nbr_ids]  # fixed shape (MAX_NBR,)

                # weight: here 1.0, replace by kernel(r)
                w = jnp.ones_like(rho_nbr)

                # mask out invalid neighbours
                contrib = jnp.where(mask, rho_nbr * w, 0.0)

                return contrib.sum()

            return jax.vmap(per_particle)(jnp.arange(self.n_particles))

        TtT_all = jax.vmap(single_TtT)(jnp.arange(n_obs))
        TtT = jnp.mean(TtT_all, axis=0)

        if lambda_smooth > 0.0:
            TtT += lambda_smooth * tikhonov_like_3d_full(x, self.m_part)

        return TtT

    # # ------------------------------------------------------------------
    # #  v-mapped ATAx  (one lane = one observation)
    # # ------------------------------------------------------------------
    # @partial(jax.jit, static_argnums=(0, 3, 4, 5))
    # def ATAx(self, x, vol_index, n_obs, lambda_smooth, distortion_model):
    #     """
    #     Vectorised over observations with jax.vmap.
    #     Computes:  (1 / n_obs) Σ_k  T_kᵀ T_k x  +  λ Δx
    #     Faster than the scan version when memory allows.
    #     """
    #
    #     # --------------------------------------------------------------
    #     # 1.  Build stencils for *all* observations in one vmap pass
    #     # --------------------------------------------------------------
    #     def build_stencil(k):
    #         xp = distortion_model.apply(self.xp_base, k, vol_index)
    #         return self.stencil_build_fn(xp)  # → (N,S) idx, (N,S) w
    #
    #     idx_all, w_all = jax.vmap(build_stencil)(jnp.arange(n_obs))
    #
    #     # shapes:  idx_all (n_obs, N, S),  w_all (n_obs, N, S)
    #
    #     # --------------------------------------------------------------
    #     # 2.  Per-observation Tᵀ T x  (one lane)
    #     # --------------------------------------------------------------
    #     def single_ttT(idx, w):
    #         # forward ρ → y  (particles ➜ cells)
    #         val = (x[:, None] * w).reshape(-1)  # (N*S,)
    #         idx_flat = idx.reshape(-1)
    #         y_cells = lax.scatter_add(
    #             jnp.zeros(self.n_cells, x.dtype),
    #             idx_flat[:, None],
    #             val,
    #             dimension_numbers=lax.ScatterDimensionNumbers(
    #                 update_window_dims=(),
    #                 inserted_window_dims=(0,),
    #                 scatter_dims_to_operand_dims=(0,)
    #             )
    #         )
    #
    #         # adjoint   y → ρ  (cells ➜ particles)
    #         gathered = y_cells[idx_flat]
    #         contrib = gathered * w.reshape(-1)
    #         part_idx = jnp.repeat(jnp.arange(self.n_particles), w.shape[1])
    #         yt = lax.scatter_add(
    #             jnp.zeros_like(x),
    #             part_idx[:, None],
    #             contrib,
    #             dimension_numbers=lax.ScatterDimensionNumbers(
    #                 update_window_dims=(),
    #                 inserted_window_dims=(0,),
    #                 scatter_dims_to_operand_dims=(0,)
    #             )
    #         )
    #         return yt  # (N_particles,)
    #
    #     # jax.lax.select_and_gather_add_p
    #     # --------------------------------------------------------------
    #     # 3.  Batch the kernel over k with vmap
    #     # --------------------------------------------------------------
    #     TtT_all = jax.vmap(single_ttT)(idx_all, w_all)  # (n_obs, N)
    #     TtT = jnp.mean(TtT_all, axis=0)  # average over obs
    #
    #     # --------------------------------------------------------------
    #     # 4.  Smoothness term
    #     # --------------------------------------------------------------
    #     if lambda_smooth > 0.0:
    #         TtT += lambda_smooth * tikhonov_like_3d_full(x, self.m_part)
    #
    #     return TtT


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


@partial(jax.jit, static_argnums=(3,))
def pic_forward(rho, idx, w, n_cells):
    val = (rho[:, None] * w).reshape(-1)     # (N*S,)
    idx_flat = idx.reshape(-1)               # (N*S,)
    return jax.ops.segment_sum(val, idx_flat, n_cells)


def pic_normal_matvec(rho: jnp.ndarray,
                          xp_base: jnp.ndarray,
                          omega: jnp.ndarray,
                          m_recon: jnp.ndarray,
                          m_distorted: jnp.ndarray,
                          n_cells: int,
                          lambda_s: float):
    """Matrix–vector product  A rho  with
           A = Σ_k T_kᵀ T_k / n_obs  +  λ_s ∇ᵀ∇  (here laplace)"""

    def single_TtT(x, idx, w):
        y = pic_forward(x, idx, w, n_cells)  # (N,)
        yt = pic_adjoint(y, idx, w)  # (n_cells,)
        return yt


    def scan_body(carry, args):
        xp = args
        idx, w = build_pic_stencils_3d(omega, m_distorted, m_recon, xp)
        out = single_TtT(rho, idx, w)
        return carry + out, None

    TtT_sum, _ = jax.lax.scan(scan_body, jnp.zeros_like(rho),
                              (xp_base))
    TtT = TtT_sum / xp_base.shape[0]

    if lambda_s > 0:
        lap_term = tikhonov_like_3d_full(rho, m_recon)
        TtT += lambda_s * lap_term

    return TtT


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
    os.makedirs(logdir, exist_ok=True)
    jax.profiler.start_trace(logdir)
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
    jax.block_until_ready(recon)

    # Stop trace recording
    jax.profiler.stop_trace()