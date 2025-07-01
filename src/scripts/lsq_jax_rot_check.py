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
from typing import Tuple

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_tracing=true"





import torch
import jax
import jax.numpy as jnp
from jax import lax
import jax.profiler
from jax.experimental import sparse
from jax.scipy.sparse.linalg import cg
from jax.scipy.optimize import minimize
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


################################################################################
# Helper – cell‑centred grid (JAX version)                                      #
################################################################################

def get_cell_centered_grid(
    omega: jnp.ndarray,          # shape (2*d,) – [x0,x1,y0,y1,...]
    m: Tuple[int, ...] | jnp.ndarray,  # voxels per dim
    return_all: bool = False,
    dtype=jnp.float32,
):
    """Return cell‑centred coordinates in physical space (JAX).

    * If *return_all* is **False** → returns the **distortion dimension** grid
      (i.e. last spatial axis): shape (N, 1).
    * If *True* → concatenates grids of *all* dims: shape (d*N, 1).

    Matches Torch implementation in utils.py.
    """
    m = jnp.asarray(m, dtype=jnp.int32)
    d = m.size
    h = (omega[1::2] - omega[0::2]) / m.astype(omega.dtype)

    def xi(i):
        start = omega[2*i]   + 0.5 * h[i]
        stop  = omega[2*i+1] - 0.5 * h[i]
        return jnp.linspace(start, stop, int(m[i]), dtype=dtype)

    if d == 1:
        return xi(0)

    grids = [xi(i) for i in range(d)]
    mesh = jnp.meshgrid(*grids, indexing="ij")  # list of (d) arrays

    if not return_all:
        # distortion (phase‑encode) dim = last axis → mesh[-1]
        return mesh[-1].reshape(-1, 1)

    # concatenate all dimensions row‑wise
    flat = [g.reshape(-1, 1) for g in mesh]
    return jnp.concatenate(flat, axis=1)


def build_pic_stencils_3d(
        omega,              # (z0,z1,y0,y1,x0,x1)
        m_source,           # (Dz, Dy, Dx)
        m_target,           # (Pz, Py, Px)
        xp,                 # (3, Np)  order: (z,y,x)
        return_jacobian=False
    ):
    Dz, Dy, Dx = m_source
    Pz, Py, Px = m_target
    n_cells     = Dz * Dy * Dx
    n_particles = Pz * Py * Px

    extent      = omega[1::2] - omega[0::2]          # (Lz,Ly,Lx)
    cell_sz     = extent / jnp.array(m_source)       # Δz,Δy,Δx
    part_sz     = extent / jnp.array(m_target)       # δz,δy,δx

    # particle positions in voxel space
    z_vox = (xp[0] - omega[0]) / part_sz[0]
    y_vox = (xp[1] - omega[2]) / part_sz[1]
    x_vox = (xp[2] - omega[4]) / part_sz[2]

    Pz_idx, wz = jnp.floor(z_vox).astype(int), z_vox - jnp.floor(z_vox)
    Py_idx, wy = jnp.floor(y_vox).astype(int), y_vox - jnp.floor(y_vox)
    Px_idx, wx = jnp.floor(x_vox).astype(int), x_vox - jnp.floor(x_vox)

    Pz_idx, Py_idx, Px_idx, wz, wy, wx = map(
        lambda a: a.reshape(-1),
        (Pz_idx, Py_idx, Px_idx, wz, wy, wx)
    )

    epsP   = 1 * part_sz
    pwidth = jnp.ceil(epsP / cell_sz).astype(int)     # (pz,py,px)

    Bz = int1DSingle(wz, int(pwidth[0]), epsP[0], cell_sz[0], part_sz[0])
    By = int1DSingle(wy, int(pwidth[1]), epsP[1], cell_sz[1], part_sz[1])
    Bx = int1DSingle(wx, int(pwidth[2]), epsP[2], cell_sz[2], part_sz[2])

    nz, ny, nx = map(lambda b: b.shape[0], (Bz, By, Bx))
    stencil_sz = nz * ny * nx

    Ilist, Jlist, Blist = [], [], []
    pp = 0
    for iz, dz in enumerate(range(-int(pwidth[0]), int(pwidth[0])+1)):
        for iy, dy in enumerate(range(-int(pwidth[1]), int(pwidth[1])+1)):
            for ix, dx in enumerate(range(-int(pwidth[2]), int(pwidth[2])+1)):

                # neighbour voxel indices
                z_idx = Pz_idx + dz
                y_idx = Py_idx + dy
                x_idx = Px_idx + dx

                # linear index
                Iijk = (z_idx * Dy + y_idx) * Dx + x_idx

                Bij  = Bz[iz] * By[iy] * Bx[ix]

                Ilist.append(Iijk)
                Jlist.append(jnp.arange(n_cells))
                Blist.append(Bij)
                pp += 1

    I = jnp.stack(Ilist, 0).reshape(stencil_sz, n_particles).T
    B = jnp.stack(Blist, 0).reshape(stencil_sz, n_particles).T

    if return_jacobian:
        Jac = jnp.zeros(n_cells, B.dtype).at[
            jnp.stack(Jlist,0).reshape(-1)
        ].add(jnp.stack(Blist,0).reshape(-1))
        return I, B, Jac
    return I, B


def B_single(x: jnp.ndarray, eps: float, h: float):
    """
    Scalar antiderivative of the PIC kernel (no derivative branch).
    Matches the torch version line-for-line.
    """
    Bij   = jnp.zeros_like(x)
    thresh = eps / h

    ind1 = (-thresh <= x) & (x <= 0)
    ind2 = (0 < x)       & (x <= thresh)
    ind3 = x > thresh

    Bij = Bij.at[ind1].set(x[ind1] + 0.5 * (h/eps) * x[ind1]**2 + eps/(2*h))
    Bij = Bij.at[ind2].set(x[ind2] - 0.5 * (h/eps) * x[ind2]**2 + eps/(2*h))
    Bij = Bij.at[ind3].set(eps / h)
    return Bij / eps

# @partial(jax.jit, static_argnums=(1,2,3,4))
def int1DSingle(w: jnp.ndarray,
                    pwidth: int,
                    eps:    float,
                    h:      float,
                    hp:     float):
    """
    jnp implementation of `int1DSingle` (without derivatives).
    Returns Bij  with shape  (2*pwidth+1, N)
    """
    N   = w.shape[0]
    Bij = jnp.zeros((2 * pwidth + 1, N), dtype=w.dtype)

    # Bleft at left edge of first bin
    Bleft = B_single(-pwidth - w, eps, h)
    for p in range(-pwidth, pwidth + 1):
        idx     = p + pwidth
        Bright  = B_single(p + 1 - w, eps, h)
        Bij     = Bij.at[idx].set(hp * (Bright - Bleft))
        Bleft   = Bright
    return Bij

################################################################################
# 3.  PIC forward / adjoint / normal matvec                                     #
################################################################################


@jax.jit
def pic_adjoint(y: jnp.ndarray,
                indices: jnp.ndarray,
                weights: jnp.ndarray) -> jnp.ndarray:
    """Compute rho = Tᵗ y from cell-space y."""
    contrib = y[indices] * weights       # shape: (N_particles, K)
    return jnp.sum(contrib, axis=1)      # shape: (N_particles,)


@partial(jax.jit, static_argnums=(3))
def pic_forward(rho, indices, weights, n_cells):
    idx = indices.reshape(-1)
    val = (rho[:, None] * weights).reshape(-1)
    # return lax.scatter_add(
    #     jnp.zeros(n_cells, dtype=val.dtype),
    #     jnp.expand_dims(idx, axis=1),
    #     val,
    #     dimension_numbers=lax.ScatterDimensionNumbers(
    #         update_window_dims=(),            # scalar updates
    #         inserted_window_dims=(0,),        # each update is inserted into a scalar slot
    #         scatter_dims_to_operand_dims=(0,) # map idx[*, 0] to output dim 0
    #     ),
    #     indices_are_sorted=False,
    #     unique_indices=False
    # )
    return jnp.zeros(n_cells).at[idx].add(val)

@partial(jax.jit, static_argnums=(3,4,5))
def pic_normal_matvec_jit(rho: jnp.ndarray,
                      indices: jnp.ndarray,
                      weights: jnp.ndarray,
                      n_cells: int,
                      lambda_s: float,
                      spatial_size):
    """Matrix–vector product  A rho  with
           A = Σ_k T_kᵀ T_k / n_obs  +  λ_s ∇ᵀ∇  (here laplace)"""

    n_obs = indices.shape[0]  # observations per slice
    n_particles = rho.shape[-1]

    def single_TtT(x, idx, w):
        y = pic_forward(x, idx, w, n_cells)             # (N,)
        yt = pic_adjoint(y, idx, w) # (n_cells,)
        return yt

    # TtT = jax.vmap(single_TtT, in_axes=(None, 0, 0))(rho, indices,
    #                                                  weights).mean(0)

    def scan_body(carry, args):
        idx, w = args
        out = single_TtT(rho, idx, w)
        return carry + out, None

    TtT_sum, _ = jax.lax.scan(scan_body, jnp.zeros_like(rho),
                              (indices, weights))
    TtT = TtT_sum / indices.shape[0]

    if lambda_s > 0:
        lap_term = tikhonov_like_3d_full(rho, spatial_size)
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
# 5.  CG solver (matrix‑free)                                                   #
################################################################################

def pcg(Ax, b, M_inv, x0, tol=1e-6, maxiter=50):
    """Pre‑conditioned CG written with `jax.lax.while_loop` for jit."""

    def body_fun(state):
        k, x, r, z, p, rz_old = state
        Ap = Ax(p)
        alpha = rz_old / jnp.vdot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M_inv(r_new)
        rz_new = jnp.vdot(r_new, z_new)
        beta = rz_new / rz_old
        p_new = z_new + beta * p
        k_new = k + 1
        return k_new, x_new, r_new, z_new, p_new, rz_new

    def cond_fun(state):
        k, x, r, *_ = state
        return jnp.logical_and(k < maxiter, jnp.linalg.norm(r) > tol)

    r0 = b - Ax(x0)
    z0 = M_inv(r0)
    state0 = (0, x0, r0, z0, z0, jnp.vdot(r0, z0))
    *_, x_final, _, _, _, _ = lax.while_loop(cond_fun, body_fun, state0)
    return x_final

################################################################################
# 6.  Public API:  solve_lsq_slice                                              #
################################################################################

@partial(jax.jit, static_argnums=(1))
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

@partial(jax.jit, static_argnums=(1,2,6))
def solve_one_vol(vol_obs,  # (n_obs, H, W)  – PE/RPE stack
                  m_recon,
                  m_distorted,  # (2,) – distorted & recon grids
                  indices,
                  weights,
                  diag,
                  lambda_smooth=0.1):
    """
    Runs the PIC-LSQ solver for *one* (volume, slice) pair.
    """
    y = vol_obs.reshape(vol_obs.shape[0], -1)
    rho_init = jnp.zeros(m_recon[-3:], dtype=y.dtype).reshape(-1)


    n_particles = m_recon[0]*m_recon[1]*m_recon[2]
    n_cells = m_distorted[0]*m_distorted[1]*m_distorted[2]


    # M_inv = lambda r: r / diag
    M_inv = lambda r: r

    Ax = lambda x: pic_normal_matvec_jit(x, indices, weights, n_cells,
                                     lambda_smooth, m_recon)

    # RHS: 1/n_obs Σ_k T_kᵀ y_k
    rhs = jnp.zeros(n_particles)
    for k in range(y.shape[0]):
        rhs += pic_adjoint(y[k], indices[k], weights[k])
    rhs /= y.shape[0]

    rho_est, _ = cg(Ax, rhs, M=M_inv, x0=rho_init, tol=1e-8, maxiter=10)
    loss = ((rho_est-rhs)**2).sum()

    return rho_est, loss


# @jax.jit
def batch_solve(observations, omega_recon, m_recon, m_distorted, xp):

    n_cells = m_distorted[0]*m_distorted[1]*m_distorted[2]
    n_particles = m_recon[0]*m_recon[1]*m_recon[2]

    xp_flat = xp.reshape(*xp.shape[0:2], -1)
    obs_res = observations.transpose(1,0,2,3,4)
    idx_all = []
    w_all = []

    for xp_grid in xp_flat:

        idx, w = build_pic_stencils_3d(omega_recon, m_recon, m_distorted, xp_grid)
        idx_all.append(idx)
        w_all.append(w)

    w = jnp.stack(w_all)
    idx = jnp.stack(idx_all)

    lambda_smooth = 0.1
    diag = diag_precond(idx, w, n_particles) + lambda_smooth

    solve_fn = lambda obs: solve_one_vol(obs, m_recon, m_distorted,
                                         idx, w, diag, lambda_smooth=lambda_smooth)
    def scan_body(carry, obs):
        result = solve_fn(obs)  # could be tuple (rho_est, residuals)
        return carry, result     # carry stays unused, just return result

    _, outputs = jax.lax.scan(scan_body, None, obs_res)

    return outputs  # shape: (N, ...) — depending on what solve_fn returns

def bc_to_xp(bc, xp_base, data, target_res):
    # Recompute the shifted, rotated particle grid from bc

    ref_mat = jnp.array(data.mats[0].numpy())
    xp_lins = []
    for pair_index, pair in enumerate(data.image_pairs):

        roi_mat = jnp.array(data.mats[pair_index].numpy())
        T_mat_permuted = jnp.linalg.inv(roi_mat) @ ref_mat

        xp_pe = xp_base + pair[0].phase_sign * bc
        # xp_pe = xp_base
        xp_lin = xp_pe.reshape(3, -1)
        ones = jnp.ones((1, xp_lin.shape[1]))
        xp_lin = jnp.vstack([xp_lin, ones])
        xp_lin = T_mat_permuted @ xp_lin
        xp_lin = xp_lin[:3].reshape(3, *target_res)
        xp_lin.at[0].set(xp_lin[0] / jnp.array(data.omega[3]))
        xp_lin.at[1].set(xp_lin[1] / jnp.array(data.omega[5]))
        xp_lin.at[2].set(xp_lin[2] / jnp.array(data.omega[7]))
        xp_lins.append(xp_lin)

        # print(T_mat_permuted)

        xp_rpe = xp_base + pair[1].phase_sign * bc
        # xp_rpe = xp_base
        xp_lin = xp_rpe.reshape(3, -1)
        ones = jnp.ones((1, xp_lin.shape[1]))
        xp_lin = jnp.vstack([xp_lin, ones])
        xp_lin = T_mat_permuted @ xp_lin
        xp_lin = xp_lin[:3].reshape(3, *target_res)
        xp_lin.at[0].set(xp_lin[0] / jnp.array(data.omega[3]))
        xp_lin.at[1].set(xp_lin[1] / jnp.array(data.omega[5]))
        xp_lin.at[2].set(xp_lin[2] / jnp.array(data.omega[7]))
        xp_lins.append(xp_lin)

    xp_lins = jnp.stack(xp_lins)

    return xp_lins


def loss_and_grad(bc, xp_base, vols, omega, target_res, m_distorted, data):
    def loss_fn(bc):
        # Convert bc into spatially-shifted coordinates
        xp = bc_to_xp(bc, xp_base, data, target_res)

        # Run the forward solver (reconstruction)
        recon, loss = batch_solve(vols, omega, target_res, m_distorted, xp)

        return loss.mean(), recon

    return jax.value_and_grad(loss_fn, has_aux=True)(bc)

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


def total_loss_fn(bc_3d_flat, unflatten_fn):
    bc_3d = unflatten_fn(bc_3d_flat)
    (data_term, _), _ = loss_and_grad(bc_3d, ...)
    lap = laplacian_3d(bc_3d)
    smooth_term = jnp.sum(lap ** 2)
    return data_term + lambda_bc * smooth_term


def make_loss_fn(unflatten_fn, xp_base, vols, omega, target_res, m_distorted, data):
    def loss_fn(flat_bc):
        bc_3d = unflatten_fn(flat_bc)
        (data_term, _), _ = loss_and_grad(bc_3d, xp_base, vols, omega, target_res, m_distorted, data)
        lap = laplacian_3d(bc_3d)
        smooth_term = jnp.sum(lap ** 2)
        return data_term + lambda_bc * smooth_term

    return loss_fn

# def make_loss_fn(xp_base, vols, omega, target_res, m_distorted, data):
#     def loss_fn(bc_3d):
#         (data_term, _), _ = loss_and_grad(bc_3d, xp_base, vols, omega, target_res, m_distorted, data)
#         lap = laplacian_3d(bc_3d)
#         smooth_term = jnp.sum(lap ** 2)
#         return data_term + lambda_bc * smooth_term
#
#     return loss_fn

def smooth_loss_fn(x):
    lap = laplacian_3d(x)
    return jnp.sum(lap ** 2)


def total_loss(bc, xp_base, vols, omega, target_res, m_distorted, data):
    xp = bc_to_xp(bc, xp_base, data, target_res)
    recon, residuals = batch_solve(vols, omega, target_res, m_distorted, xp)
    data_term = jnp.sum(residuals ** 2)
    smooth_term = jnp.sum(laplacian_3d(bc) ** 2)
    return data_term + lambda_bc * smooth_term

if __name__ == "__main__":
    import numpy as np

    # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # logdir = f"/home/laurin/workspace/PyHySCO/data/results/debug/tensor_logs/{timestamp}"
    # os.makedirs(logdir, exist_ok=True)
    # jax.profiler.start_trace(logdir)
    start_time = time.time()

    image_config_file = "/home/laurin/workspace/PyHySCO/data/raw/lowres/image_config.json"
    device =  'cpu'
    pair_idx = [0,1,2,3]
    # pair_idx = 0
    data = MultiPeDtiData(image_config_file, device=device, dtype=torch.float32, pair_idx=pair_idx)

    target_res = [*data.m[:-2], 66, 66]
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

    # T_world = jnp.array(data.mats[0])
    # xp_base = xp_base.reshape(3, -1)
    # ones = jnp.ones((1, xp_base.shape[1]))
    # xp_base = jnp.vstack([xp_base, ones])
    # xp_base_world = T_world @ xp_base
    # xp_base_world = xp_base_world[0:3]


    vols = []
    for pair_index, pair in enumerate(data.image_pairs):
        pe, rpe = pair
        vols.append(jnp.array(pe.data.numpy()))
        vols.append(jnp.array(rpe.data.numpy()))

    dwi_images = jnp.stack(vols)
    n_obs, n_vol, D, H, W = dwi_images.shape


    target_res_tuple = (
    target_res[1].item(), target_res[2].item(), target_res[3].item())
    m_distorted_tuple = (
    m_distorted[1].item(), m_distorted[2].item(), m_distorted[3].item())
    lambda_bc = 10

    # bc_flat, unflatten = jax.flatten_util.ravel_pytree(bc_3d)

    omega_3d = jnp.array(omega[2:], dtype=jnp.float32)
    # loss_fn = make_loss_fn(unflatten, xp_base, dwi_images, omega_3d, target_res_tuple, m_distorted_tuple, data)
    #
    # # Optional: flatten bc if needed
    # x0 = bc_flat  # can be any JAX array or pytree
    # opt_options = dict(maxiter=5)
    # result = minimize(loss_fn, x0, method="BFGS", options=opt_options)  # or "L-BFGS-B", etc.
    #
    # bc_optimized = result.x
    # bc_optimized = bc_optimized.reshape(3, 78, 25, 66, 66)
    #
    # save_jax_data(bc_optimized.transpose(3, 2, 1, 0),
    #                   f"/home/laurin/workspace/PyHySCO/data/results/debug/bc_3d_opt.nii.gz")


    opt = optax.adam(learning_rate=0.5)

    # bc_flat, unflatten = jax.flatten_util.ravel_pytree(bc_3d)
    opt_state = opt.init(bc_3d)
    num_steps = 1
    # loss_fn = make_loss_fn(unflatten, xp_base, dwi_images, omega_3d, target_res_tuple, m_distorted_tuple, data)

    for step in range(num_steps):
        (data_term, recon), data_grad = loss_and_grad(bc_3d, xp_base, dwi_images, omega_3d, target_res_tuple, m_distorted_tuple, data)

        # lap = laplacian_3d(bc_3d)
        smooth_term = smooth_loss_fn(bc_3d)
        smooth_grad = jax.grad(smooth_loss_fn)(bc_3d)

        total_grad = data_grad + lambda_bc * smooth_grad

        total_loss = data_term + lambda_bc * smooth_term

        updates, opt_state = opt.update(total_grad, opt_state, params=bc_3d)

        bc_3d = optax.apply_updates(bc_3d, updates)

        print(f"{step}: loss = {total_loss:.4e}")

        save_jax_data(recon.reshape(*target_res).transpose(3, 2, 1, 0),
                      f"/home/laurin/workspace/PyHySCO/data/results/debug/recon_{step}.nii.gz")
        save_jax_data(total_grad.transpose(3, 2, 1, 0),
                      f"/home/laurin/workspace/PyHySCO/data/results/debug/grad_{step}.nii.gz")
        save_jax_data(bc_3d.transpose(3, 2, 1, 0),
                      f"/home/laurin/workspace/PyHySCO/data/results/debug/bc_3d_{step}.nii.gz")
        save_jax_data(smooth_grad.transpose(3, 2, 1, 0),
                      f"/home/laurin/workspace/PyHySCO/data/results/debug/grad_smooth_{step}.nii.gz")
        save_jax_data(data_grad.transpose(3, 2, 1, 0),
                      f"/home/laurin/workspace/PyHySCO/data/results/debug/grad_data_{step}.nii.gz")


