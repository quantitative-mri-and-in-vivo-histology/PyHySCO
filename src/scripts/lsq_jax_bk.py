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

# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"



import torch
import jax
import jax.numpy as jnp
from jax import lax
import jax.profiler
import os
import time

from triton.backends.nvidia.compiler import min_dot_size

jax.config.update("jax_disable_jit", True)

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



# ###############################################################################
# # GLOBAL STATICS                                                              #
# ###############################################################################
# MAX_PWIDTH = 3                            # largest half‑width you will need
# K1         = 2 * MAX_PWIDTH + 1           # stencil size in 1‑D
# OFF_1D     = jnp.arange(-MAX_PWIDTH, MAX_PWIDTH + 1, dtype=jnp.int32)  # (K1,)
# ###############################################################################
# # 0. 1‑D basis (vectorised, static shape, tracer‑safe)                         #
# ###############################################################################
#
# def _B_single(x: jnp.ndarray, eps: float, h: float) -> jnp.ndarray:
#     thresh = eps / h
#     val1   = x + 0.5 * (h / eps) * x ** 2 + eps / (2 * h)   # −th ≤ x ≤ 0
#     val2   = x - 0.5 * (h / eps) * x ** 2 + eps / (2 * h)   # 0 < x ≤ th
#     val3   = eps / h                                        # x > th
#     out    = jnp.where(x <= 0, val1, val2)
#     out    = jnp.where(x > thresh, val3, out)
#     return out / eps
#
#
# def _int1d_single(w: jnp.ndarray,
#                   pwidth: jnp.ndarray,    # may be tracer scalar
#                   eps: float,
#                   h: float,
#                   hp: float) -> jnp.ndarray:
#     """Return weights of **static** shape (K1, N) regardless of pwidth."""
#     off   = OFF_1D[:, None]                    # (K1,1)
#     Bij   = hp * (_B_single(off + 1 - w, eps, h) -
#                   _B_single(off     - w, eps, h))      # (K1,N)
#     mask  = (jnp.abs(off) <= pwidth)            # (K1,1)
#     return jnp.where(mask, Bij, 0.0)
#
# ###############################################################################
# # 1. build_pic_stencils_2d  (static K, tracer‑safe)                            #
# ###############################################################################
#
# def build_pic_stencils_2d(
#     omega: jnp.ndarray,               # [x0,x1,y0,y1]
#     m_src: Tuple[int, int],           # (Hs, Ws)
#     m_tgt: Tuple[int, int],           # (Ht, Wt)
#     xp: jnp.ndarray,                  # (2, N) phys coords
#     *,
#     return_jacobian: bool = False,
# ):
#     Hs, Ws = m_src
#     Ht, Wt = m_tgt
#     n_cells = Hs * Ws
#     n_part = m_tgt[0] * m_tgt[1]
#     # concrete
#     # Ht, Wt = m_tgt
#     # n_cells = Hs * Ws
#     # n_part  = Ht * Wt
#
#     cell_sz = (omega[1::2] - omega[0::2]) / jnp.array([Hs, Ws])
#     part_sz = (omega[1::2] - omega[0::2]) / jnp.array([Ht, Wt])
#     epsP    = part_sz
#
#     # dynamic pwidth (scalar tracers)
#     pwx = jnp.ceil(epsP[0] / cell_sz[0]).astype(jnp.int32)
#     pwy = jnp.ceil(epsP[1] / cell_sz[1]).astype(jnp.int32)
#
#     # particle → voxel coords
#     x_vox = (xp[0] - omega[0]) / part_sz[0]
#     y_vox = (xp[1] - omega[2]) / part_sz[1]
#     Px    = jnp.floor(x_vox).astype(jnp.int32)
#     Py    = jnp.floor(y_vox).astype(jnp.int32)
#
#     # Build full static offsets grid (K = K1*K1)
#     dx, dy = jnp.meshgrid(OFF_1D, OFF_1D, indexing="ij")
#     dx = dx.reshape(-1)   # (K,)
#     dy = dy.reshape(-1)
#     K  = dx.size
#
#     # indices
#     idx_x   = Px[:, None] + dx          # (N,K)
#     idx_y   = Py[:, None] + dy
#     indices = idx_y * Ht + idx_x        # row‑major (N,K)
#
#     # weights – separable product
#     wx = x_vox - Px
#     wy = y_vox - Py
#     Bx = _int1d_single(wx[None, :], pwx, epsP[0], cell_sz[0], part_sz[0])  # (K1,N)
#     By = _int1d_single(wy[None, :], pwy, epsP[1], cell_sz[1], part_sz[1])  # (K1,N)
#     Bx = Bx.reshape(K1, -1)      # (K1,N)
#     By = By.reshape(K1, -1)
#     weights = (Bx[:, None, :] * By[None, :, :]).reshape(K, -1).T  # (N,K)
#
#     # T = Bx[0].block_until_ready()
#     # Tz = Bx[:,0].block_until_ready()
#     # # jax.debug.print(T)
#
#     jax.debug.print("Bx[:,0] = {}", Bx[:,0])
#
#     # mask invalid voxel ids
#     valid   = (indices >= 0) & (indices < n_part)
#     weights = jnp.where(valid, weights, 0.0)
#     indices = jnp.where(valid, indices, 0)
#
#     if not return_jacobian:
#         return indices, weights
#
#     Jac = jnp.sum(weights, axis=1)
#     return indices, weights, Jac


def build_pic_stencils_2d(omega, m_source, m_target, xp, return_jacobian=False):
    Hs, Ws = m_source
    Ht, Wt = m_target
    n_cells = Hs * Ws
    n_particles = Ht * Wt

    cell_size = (omega[1::2] - omega[0::2]) / jnp.array(m_source)
    particle_sz = (omega[1::2] - omega[0::2]) / jnp.array(m_target)

    x_vox = (xp[0] - omega[0]) / particle_sz[0]
    y_vox = (xp[1] - omega[2]) / particle_sz[1]

    Px, wx = jnp.floor(x_vox).astype(int), x_vox - jnp.floor(x_vox)
    Py, wy = jnp.floor(y_vox).astype(int), y_vox - jnp.floor(y_vox)

    # wx = wx*0
    # wx = wx+0.74
    #
    # jax.debug.print("wx = {}", wx[0:10])
    jax.debug.print("m_source = {}", m_source)
    jax.debug.print("m_target = {}", m_target)
    jax.debug.print("omega = {}", omega)

    Px, Py, wx, wy = map(lambda a: a.reshape(-1), (Px, Py, wx, wy))

    epsP = 1 * particle_sz
    pwidth = jnp.ceil(epsP / cell_size).astype(int)
    # pwidth = [3,3]

    Bx = int1DSingle(wx, int(pwidth[0]), epsP[0], cell_size[0], particle_sz[0])
    By = int1DSingle(wy, int(pwidth[1]), epsP[1], cell_size[1], particle_sz[1])


    # Bx = Bx / Bx.sum(0)
    # By = By / By.sum(0)

    # jax.debug.print("Bx[:,0] = {}", Bx[:,0])

    nbx, nby = Bx.shape[0], By.shape[0]
    stencil_sz = nbx * nby

    I = []
    J = []
    B = []
    pp = 0
    for i, px in enumerate(range(-int(pwidth[0]), int(pwidth[0]) + 1)):
        for j, py in enumerate(range(-int(pwidth[1]), int(pwidth[1]) + 1)):
            x_idx = Px + px
            y_idx = Py + py
            Iij = y_idx * Wt + x_idx
            # Iij = x_idx * m_target[1] + y_idx
            Bij = Bx[i] * By[j]
            I.append(Iij)
            J.append(jnp.arange(n_cells))
            B.append(Bij)
            pp += 1

    I = jnp.stack(I, axis=0).reshape(stencil_sz, n_cells).T
    B = jnp.stack(B, axis=0).reshape(stencil_sz, n_cells).T
    I = jnp.where((I >= 0) & (I < n_particles), I, 0)
    B = jnp.where((I >= 0) & (I < n_particles), B, 0.0)

    if return_jacobian:
        Jac = jnp.zeros(n_cells, dtype=B.dtype).at[jnp.stack(J, axis=0).reshape(-1)].add(jnp.stack(B, axis=0).reshape(-1))
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
# 2.  Finite‑difference smoothness operator                                     #
################################################################################

def laplacian_2d(H: int, W: int):
    """Return a function  L(r) = laplace(r)  using 5‑point stencil."""
    N = H * W

    def _flat(u: jnp.ndarray) -> jnp.ndarray:
        return u.reshape(H, W)

    def _lap(u: jnp.ndarray) -> jnp.ndarray:  # u flat (N,)
        U = _flat(u)
        # zero‑Neumann boundaries
        up   = jnp.roll(U, -1, 0)
        down = jnp.roll(U,  1, 0)
        left = jnp.roll(U, -1, 1)
        right= jnp.roll(U,  1, 1)
        return (up + down + left + right - 4 * U).reshape(-1)

    return _lap

################################################################################
# 3.  PIC forward / adjoint / normal matvec                                     #
################################################################################

def pic_forward(rho: jnp.ndarray, indices: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Compute  y = T rho  using gathered indices / weights.

    rho        : (N,)           – particles (flattened)
    indices    : (N, K) int32   – voxel id per stencil weight
    weights    : (N, K) float32 – weights per stencil weight
    returns    : (N,)           – voxel values per particle (actually length N)
    """
    gathered = rho[indices]          # (N, K)
    return jnp.sum(gathered * weights, axis=-1)  # (N,)


def pic_adjoint(y: jnp.ndarray, indices: jnp.ndarray, weights: jnp.ndarray, n_particles: int) -> jnp.ndarray:
    """Compute z = Tᵀ y.

    y          : (N,)           – values at particle sites
    indices    : (N, K) int32   – voxel indices
    weights    : (N, K) float32 – same weights
    n_cells    : int
    returns    : (n_cells,)     – adjoint result on particles
    """
    contrib = y[:, None] * weights        # (N, K)
    idx = indices.reshape(-1)
    val = contrib.reshape(-1)
    z = jnp.zeros(n_particles, y.dtype)
    z = z.at[idx].add(val)
    return z


def pic_normal_matvec(rho: jnp.ndarray,
                      indices: jnp.ndarray,
                      weights: jnp.ndarray,
                      n_cells: int,
                      lambda_s: float,
                      laplace_fn):
    """Matrix–vector product  A rho  with
           A = Σ_k T_kᵀ T_k / n_obs  +  λ_s ∇ᵀ∇  (here laplace)"""

    n_obs = indices.shape[0]  # observations per slice
    n_particles = rho.shape[-1]

    def single_TtT(x, idx, w):
        y = pic_forward(x, idx, w)             # (N,)
        yt = pic_adjoint(y, idx, w, n_particles) # (n_cells,)
        return yt

    # def single_TtT(x, idx, w):
    #     y = pic_adjoint(x, idx, w, n_particles)             # (N,)
    #     return pic_forward(y, idx, w) # (n_particles,)

    TtT = jnp.zeros_like(rho)
    for k in range(n_obs):
        TtT += single_TtT(rho, indices[k], weights[k])
    TtT /= n_obs

    if lambda_s > 0:
        lap_term = laplace_fn(rho)
        TtT += lambda_s * lap_term
    return TtT

################################################################################
# 4.  Diagonal pre‑conditioner                                                  #
################################################################################

def diag_precond(indices: jnp.ndarray, weights: jnp.ndarray, n_cells: int) -> jnp.ndarray:
    """Return   diag(Tᵀ T)   to be used as a Jacobi PC."""
    w2 = jnp.sum(weights ** 2, axis=0)         # (N,)  per particle sum
    diag = jnp.zeros(n_cells)
    diag = diag.at[indices].add(w2)
    return jnp.maximum(diag, 1e-6)             # avoid divide‑by‑0

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

def tikhonov_2d(H, W):
    """
    Returns a function: r ↦ LᵀL r = (Dxᵀ Dx + Dyᵀ Dy) r
    where Dx, Dy are first-order difference operators in x and y.
    """
    def apply(r):
        U = r.reshape(H, W)
        dx = U - jnp.roll(U, 1, axis=1)  # backward difference in x
        dy = U - jnp.roll(U, 1, axis=0)  # backward difference in y

        # zero-pad boundary differences
        dx = dx.at[:, 0].set(0.0)
        dy = dy.at[0, :].set(0.0)

        # adjoint = negative backward difference
        dxt = dx - jnp.roll(dx, -1, axis=1)
        dyt = dy - jnp.roll(dy, -1, axis=0)

        # zero adjoint BCs
        dxt = dxt.at[:, -1].set(0.0)
        dyt = dyt.at[-1, :].set(0.0)

        return (dxt + dyt).reshape(-1)

    return apply

# @partial(jax.jit, static_argnums=(6))
def solve_lsq_slice(
    rho_init: jnp.ndarray,              # (n_cells,)
    observations: jnp.ndarray,          # (n_obs, n_cells)
    indices: jnp.ndarray,               # (n_obs, N, K)
    weights: jnp.ndarray,               # (n_obs, N, K)
    m_recon: jnp.ndarray,
    m_target: jnp.ndarray,
    lambda_smooth: float = 0.0001,
):
    n_particles = m_recon[0]*m_recon[1]

    lap_fn = tikhonov_2d(m_recon[0], m_recon[1])

    # Pre‑conditioner (Jacobi)
    diag = diag_precond(indices, weights, n_particles) + lambda_smooth
    M_inv = lambda r: r / diag
    # M_inv = lambda r: r

    Ax = lambda x: pic_normal_matvec(x, indices, weights, n_particles,
                                     lambda_smooth, lap_fn)

    # RHS: 1/n_obs Σ_k T_kᵀ y_k
    rhs = jnp.zeros(n_particles)
    for k in range(observations.shape[0]):
        rhs += pic_adjoint(observations[k], indices[k], weights[k], n_particles)
    rhs /= observations.shape[0]

    rho_est = pcg(Ax, rhs, M_inv, rho_init, tol=1e-5, maxiter=20)
    return rho_est

################################################################################
# 7.  Example usage                                                             #
################################################################################


def solve_one_slice(vol_slice_obs,             # (n_obs, H, W)  – PE/RPE stack
                    omega_recon,                     # length-4 array for this slice
                    m_recon,
                    m_distorted,              # (2,) – distorted & recon grids
                    xp,
                    lambda_smooth=0.1):
    """
    Runs the PIC-LSQ solver for *one* (volume, slice) pair.
    """
    # ------------------------------------------------------------------
    # 1. Build particle positions and stencils *for this slice only*
    # ------------------------------------------------------------------
    n_obs, H, W = vol_slice_obs.shape

    idx, w = build_pic_stencils_2d(omega_recon, m_distorted, m_recon, xp)
    idx = idx[None]          # add obs axis dim=1 later
    w   = w[None]

    # ------------------------------------------------------------------
    # 2. Flatten observations to (n_obs, N)
    # ------------------------------------------------------------------
    y = vol_slice_obs.reshape(vol_slice_obs.shape[0], -1)

    # ------------------------------------------------------------------
    # 3. Call the JIT-compiled solver
    # ------------------------------------------------------------------
    rho_init = jnp.zeros(y.shape[1:], dtype=y.dtype)
    rho_hat  = solve_lsq_slice(rho_init, y, idx, w, m_recon, m_distorted,
                               lambda_smooth=lambda_smooth)
    return rho_hat


# @jax.jit
def batch_solve(observations, omega_recon, m_recon, m_distorted, xp):

    # solve_fn = lambda obs: solve_one_slice(obs, omega, m_distorted, m_recon, xp)
    # return jax.vmap(  # over volumes
    #     jax.vmap(solve_fn)  # over slices
    # )(observations)

    # results = []
    # for slice_index, ob in enumerate(observations):
    #     xp_slice = xp[:,slice_index]
    #     xp_slice_flat = xp_slice.reshape(xp_slice.shape[0], -1)
    #     solve_fn = lambda obs: solve_one_slice(obs, omega, m_distorted, m_recon, xp_slice_flat)
    #     results.append(jax.vmap(solve_fn)(ob))

    results = []
    for slice_index, ob in enumerate(observations):
        xp_slice = xp[:,slice_index]
        xp_slice_flat = xp_slice.reshape(xp_slice.shape[0], -1)
        results_per_vol = []
        for vol_index, vol in enumerate(ob):
            results_per_vol.append(solve_one_slice(vol, omega_recon, m_recon, m_distorted, xp_slice_flat))
        results.append(jnp.stack(results_per_vol, axis=0))

    return jnp.stack(results)


if __name__ == "__main__":
    import numpy as np

    # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # logdir = f"/home/laurin/workspace/PyHySCO/data/results/debug/tensor_logs/{timestamp}"
    # os.makedirs(logdir, exist_ok=True)
    # jax.profiler.start_trace(logdir)
    # start_time = time.time()

    image_config_file = "/home/laurin/workspace/PyHySCO/data/raw/lowres/image_config.json"
    device =  'cpu'
    # pair_idx = [0,1,2,3]
    pair_idx = 0
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

    target_res = jnp.array(target_res.numpy())


    vols = []
    for pair in data.image_pairs:
        pe, rpe = pair
        vols.append(jnp.array(pe.data.numpy()))
        vols.append(jnp.array(rpe.data.numpy()))

    dwi_images = jnp.stack(vols)
    dwi_images = dwi_images.transpose(2, 1, 0, 3, 4)

    dwi_images = dwi_images.reshape(-1, *dwi_images.shape[1:])

    n_obs, H, W = dwi_images.shape[2:]

    m_distorted = jnp.array(data.m.numpy())

    # 3. Build stencils (identity)
    omega = data.omega[4:].numpy()
    # m_distorted = jnp.array([H, W])
    # m_recon = jnp.array([H, W])

    # m_distorted = (8, 64, 64)
    # m_recon = (8, 128, 128)

    # H, W = 128, 128
    # m_distorted = (H, W)  # tuple, not jnp.array
    # m_recon = (H, W)

    xp = get_cell_centered_grid(omega, target_res[2:], return_all=True)
    xp = xp.transpose(1,0)

    xp_offset = jnp.array(bc_2d.numpy())
    xp_offset = xp_offset.reshape(*xp_offset.shape[0:2], -1)

    xp_offset = xp_offset + jnp.tile(xp[:, None, :], (1, xp_offset.shape[1], 1))
    xp_offset = xp_offset.reshape(*xp_offset.shape[0:2], *target_res[2:])
    xp_offset = xp_offset

    # Run solver
    _, _, n_slices, H, W = dwi_images.shape
    recon = batch_solve(dwi_images,
                        omega_recon=jnp.array(data.omega[4:], dtype=jnp.float32),
                        m_recon=target_res[2:],
                        m_distorted=m_distorted[2:],
                        xp=xp_offset)

    recon = recon.reshape(*recon.shape[0:2], *target_res[-2:])
    recon = recon.transpose(2,3,0,1)
    save_jax_data(recon,
      "/home/laurin/workspace/PyHySCO/data/results/debug/jax_recon.nii.gz")

    x = 1

    # result shape: (n_vols, n_slices, H, W)

    # # 4. Run solver
    # rho = solve_lsq_slice(jnp.zeros_like(y0), observations, indices, weights, H, W)
    # rho = jnp.where(jnp.isfinite(rho), rho, 0.0)

    # save_jax_data(rho.reshape(H,W), "/home/laurin/workspace/PyHySCO/data/results/debug/jax_rho.nii.gz")

    # jax.profiler.stop_trace()
    # print(f"Trace written to {logdir} (elapsed: {time.time() - start_time:.2f}s)")