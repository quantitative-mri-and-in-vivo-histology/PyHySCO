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
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_tracing=true"





import torch
import jax
import jax.numpy as jnp
from jax import lax
import jax.profiler
from jax.experimental import sparse
from jax.scipy.sparse.linalg import cg
import os
import time

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

    I = jnp.stack(Ilist, 0).reshape(stencil_sz, n_cells).T
    B = jnp.stack(Blist, 0).reshape(stencil_sz, n_cells).T

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
    w2 = jnp.sum(weights ** 2, axis=[0,2])         # (N,)  per particle sum
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

@partial(jax.jit, static_argnums=(4,5,7))
def solve_lsq_vol(
    rho_init: jnp.ndarray,              # (n_cells,)
    observations: jnp.ndarray,          # (n_obs, n_cells)
    indices: jnp.ndarray,               # (n_obs, N, K)
    weights: jnp.ndarray,               # (n_obs, N, K)
    m_recon: jnp.ndarray,
    m_distorted: jnp.ndarray,
    diag,
    lambda_smooth: float = 0.01,
):
    n_particles = m_recon[0]*m_recon[1]*m_recon[2]
    n_cells = m_distorted[0]*m_distorted[1]*m_distorted[2]

    # lap_fn = tikhonov_like_3d(m_recon[0], m_recon[1], m_recon[2])

    # Pre‑conditioner (Jacobi)

    M_inv = lambda r: r / diag
    # M_inv = lambda r: r

    Ax = lambda x: pic_normal_matvec_jit(x, indices, weights, n_cells,
                                     lambda_smooth, m_recon)

    # RHS: 1/n_obs Σ_k T_kᵀ y_k
    rhs = jnp.zeros(n_particles)
    for k in range(observations.shape[0]):
        rhs += pic_adjoint(observations[k], indices[k], weights[k])
    rhs /= observations.shape[0]

    # rho_est = pcg(Ax, rhs, M_inv, rho_init, tol=1e-8, maxiter=10)
    rho_est, _ = cg(Ax, rhs, M=M_inv, x0=rho_init, tol=1e-8, maxiter=10)
    return rho_est

################################################################################
# 7.  Example usage                                                             #
################################################################################

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


    M_inv = lambda r: r / diag
    # M_inv = lambda r: r

    Ax = lambda x: pic_normal_matvec_jit(x, indices, weights, n_cells,
                                     lambda_smooth, m_recon)

    # RHS: 1/n_obs Σ_k T_kᵀ y_k
    rhs = jnp.zeros(n_particles)
    for k in range(y.shape[0]):
        rhs += pic_adjoint(y[k], indices[k], weights[k])
    rhs /= y.shape[0]

    rho_est, _ = cg(Ax, rhs, M=M_inv, x0=rho_init, tol=1e-8, maxiter=10)
    return rho_est


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

    batch_size = 12
    N = obs_res.shape[0]
    solve_fn = lambda obs: solve_one_vol(obs, m_recon, m_distorted,
                                          idx, w, diag, lambda_smooth=lambda_smooth)
    results = []
    for i in range(0, N, batch_size):
        chunk = obs_res[i:i + batch_size]
        result = jax.vmap(solve_fn)(chunk)  # only apply vmap to a smaller chunk
        results.append(result)
    results = jnp.concatenate(results, axis=0)

    return results


if __name__ == "__main__":
    import numpy as np

    # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # logdir = f"/home/laurin/workspace/PyHySCO/data/results/debug/tensor_logs/{timestamp}"
    # os.makedirs(logdir, exist_ok=True)
    # jax.profiler.start_trace(logdir)
    start_time = time.time()

    image_config_file = "/home/laurin/workspace/PyHySCO/data/raw/highres/image_config.json"
    device =  'cpu'
    pair_idx = [0,1,2,3]
    # pair_idx = 0
    data = MultiPeDtiData(image_config_file, device=device, dtype=torch.float32, pair_idx=pair_idx)

    target_res = [*data.m[:-2], 128, 128]
    target_res = torch.tensor(target_res, dtype=torch.int32, device=device)
    initialization = InitializeCFMultiePeDtiDataResampled()
    B0 = initialization.eval(data, target_res, blur_result=True)

    avg_op = myAvg1D(data.omega[2:], target_res[1:], device=device, dtype=torch.float32)
    # B0_res = B0.reshape(-1,1)
    bc = avg_op.mat_mul(B0).reshape(-1,1)

    v_pe = torch.tensor([0.0, 0.0, 1.0], device=device,
                             dtype=torch.float32)
    bc_3d = (bc * v_pe.view(1, -1)).T  # shift vector in original space
    bc_3d = bc_3d.reshape(3, *target_res[1:])
    bc_2d = bc_3d[1:]
    bc_3d = jnp.array(bc_3d)

    target_res = jnp.array(target_res.numpy())
    m_distorted = jnp.array(data.m.numpy())
    omega = data.omega.numpy()

    xp_base = get_cell_centered_grid(omega[2:], target_res[1:], return_all=True)
    xp_base = xp_base.transpose(1,0)
    xp_base = xp_base.reshape(3, *target_res[-3:])

    image_center = 0.5 * (
            torch.tensor(data.omega[3::2]) + torch.tensor(
        data.omega[2::2]))  # (x_c, y_c, z_c)
    image_center = jnp.array(image_center.numpy())
    image_center_res = image_center.reshape(3,1,1,1)

    vols = []
    xp_lins = []
    for pair_index, pair in enumerate(data.image_pairs):
        pe, rpe = pair
        vols.append(jnp.array(pe.data.numpy()))
        vols.append(jnp.array(rpe.data.numpy()))

        # rot_mat_permuted = jnp.linalg.inv(data.rel_mats[pair_index][:3, :3].numpy())
        rot_mat_permuted = jnp.array(data.rel_mats[pair_index][:3, :3].numpy())


        xp_pe = xp_base + pe.phase_sign * bc_3d
        xp_lin = xp_pe.reshape(3,-1)
        xp_lin = rot_mat_permuted @ (xp_lin - image_center.reshape(-1,1)) + image_center.reshape(-1,1)
        xp_lin = xp_lin.reshape(3, *target_res[1:])
        xp_lins.append(xp_lin)

        xp_rpe = xp_base + rpe.phase_sign * bc_3d
        xp_lin = xp_rpe.reshape(3, -1)
        xp_lin = rot_mat_permuted @ (xp_lin - image_center.reshape(-1,1))  + image_center.reshape(-1,1)
        xp_lin = xp_lin.reshape(3, *target_res[1:])
        xp_lins.append(xp_lin)

    xp_lins = jnp.stack(xp_lins)

    dwi_images = jnp.stack(vols)
    n_obs, n_vol, D, H, W = dwi_images.shape

    target_res_tuple = (
    target_res[1].item(), target_res[2].item(), target_res[3].item())
    m_distorted_tuple = (
    m_distorted[1].item(), m_distorted[2].item(), m_distorted[3].item())

    # Run solver
    _, _, n_slices, H, W = dwi_images.shape
    recon = batch_solve(dwi_images,
                        omega_recon=jnp.array(omega[2:], dtype=jnp.float32),
                        m_recon=target_res_tuple,
                        m_distorted=m_distorted_tuple,
                        xp=xp_lins)

    recon = recon.reshape(recon.shape[0], *target_res[-3:])
    recon = recon.transpose(3,2,1,0)
    save_jax_data(recon,
      "/home/laurin/workspace/PyHySCO/data/results/debug/jax_recon.nii.gz")

    x = 1

    end = time.time()
    print(f"Took {end - start_time:.4f} seconds")
    # jax.profiler.stop_trace()