"""epi_lsq_jax.py  — static‑shape, tracer‑safe revision (v3)
Adds **automatic differentiation** support so you can obtain
∂J/∂ρ (or higher‑order derivatives) with `jax.grad`/`jax.jacfwd`.

Key additions
-------------
1.  `lsq_loss(rho, y, idx, w, H, W, λ)` – scalar objective
    J(ρ) = ½‖Tρ − y‖² + λ‖∇ρ‖² (batch‑averaged).
2.  `loss_and_grad = jax.value_and_grad(lsq_loss)` – returns *(J, ∇J)*.
3.  `solve_lsq_slice()` now has an optional `return_grad` flag.

Everything stays JIT‑compatible and still uses the static‑shape stencil
implementation from v2.
"""

from __future__ import annotations
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import lax

###############################################################################
# Globals (unchanged)                                                         #
###############################################################################
MAX_PWIDTH = 3
K1         = 2 * MAX_PWIDTH + 1
OFF_1D     = jnp.arange(-MAX_PWIDTH, MAX_PWIDTH + 1, dtype=jnp.int32)

###############################################################################
# 1‑D basis & stencil construction (unchanged from v2)                        #
###############################################################################

def _B_single(x: jnp.ndarray, eps: float, h: float) -> jnp.ndarray:
    thresh = eps / h
    val1   = x + 0.5 * (h / eps) * x ** 2 + eps / (2 * h)
    val2   = x - 0.5 * (h / eps) * x ** 2 + eps / (2 * h)
    val3   = eps / h
    out    = jnp.where(x <= 0, val1, val2)
    return jnp.where(x > thresh, val3, out) / eps


def _int1d_single(w: jnp.ndarray, pwidth: jnp.ndarray,
                  eps: float, h: float, hp: float) -> jnp.ndarray:
    off   = OFF_1D[:, None]                       # (K1,1)
    Bij   = hp * (_B_single(off + 1 - w, eps, h) -
                  _B_single(off     - w, eps, h))
    mask  = (jnp.abs(off) <= pwidth)
    return jnp.where(mask, Bij, 0.0)              # (K1,N)


def build_pic_stencils_2d(omega: jnp.ndarray,
                          m_src: Tuple[int, int],
                          m_tgt: Tuple[int, int],
                          xp:    jnp.ndarray,
                          *, return_jacobian: bool = False):
    """Return `(indices, weights)` with static K = K1*K1."""
    Hs, Ws = m_src
    Ht, Wt = m_tgt
    n_part = Ht * Wt
    cell_sz = (omega[1::2] - omega[0::2]) / jnp.array([Hs, Ws])
    part_sz = (omega[1::2] - omega[0::2]) / jnp.array([Ht, Wt])
    epsP    = part_sz
    pwx = jnp.ceil(epsP[0] / cell_sz[0]).astype(jnp.int32)
    pwy = jnp.ceil(epsP[1] / cell_sz[1]).astype(jnp.int32)

    x_vox = (xp[0] - omega[0]) / part_sz[0]
    y_vox = (xp[1] - omega[2]) / part_sz[1]
    Px    = jnp.floor(x_vox).astype(jnp.int32)
    Py    = jnp.floor(y_vox).astype(jnp.int32)

    dx, dy = jnp.meshgrid(OFF_1D, OFF_1D, indexing="ij")
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    idx_x   = Px[:, None] + dx
    idx_y   = Py[:, None] + dy
    indices = idx_y * Ht + idx_x                   # (N,K)

    wx = x_vox - Px
    wy = y_vox - Py
    Bx = _int1d_single(wx[None, :], pwx, epsP[0], cell_sz[0], part_sz[0])
    By = _int1d_single(wy[None, :], pwy, epsP[1], cell_sz[1], part_sz[1])
    weights = (Bx[:, None, :] * By[None, :, :]).reshape(K1*K1, -1).T

    valid   = (indices >= 0) & (indices < n_part)
    indices = jnp.where(valid, indices, 0)
    weights = jnp.where(valid, weights, 0.0)

    if not return_jacobian:
        return indices, weights
    return indices, weights, jnp.sum(weights, axis=1)

###############################################################################
# Smoothness term                                                             #
###############################################################################

def laplacian_2d(H: int, W: int):
    def _lap(u_flat):
        U = u_flat.reshape(H, W)
        return (jnp.roll(U, -1, 0) + jnp.roll(U, 1, 0) +
                jnp.roll(U, -1, 1) + jnp.roll(U, 1, 1) - 4*U).reshape(-1)
    return _lap

###############################################################################
# Forward / adjoint ops                                                       #
###############################################################################

def pic_forward(rho, idx, w):
    return jnp.sum(rho[idx] * w, axis=-1)

def pic_adjoint(y, idx, w, N):
    z = jnp.zeros(N, y.dtype)
    z = z.at[idx.reshape(-1)].add((y[:, None] * w).reshape(-1))
    return z

###############################################################################
# LSQ objective & gradient                                                    #
###############################################################################

def lsq_loss(rho_flat: jnp.ndarray,
             y_obs:    jnp.ndarray,      # (n_obs, N)
             idx:      jnp.ndarray,      # (n_obs, N, K)
             w:        jnp.ndarray,      # (n_obs, N, K)
             H: int, W: int,
             lambda_s: float = 0.0) -> jnp.ndarray:
    N      = H*W
    lap_fn = laplacian_2d(H, W)

    def single_residual(r, i, wi):
        return pic_forward(r, i, wi)            # (N,)

    # broadcast over observations
    residuals = jax.vmap(single_residual, in_axes=(None, 0, 0))(rho_flat, idx, w) - y_obs
    data_term = 0.5 * jnp.mean(jnp.sum(residuals**2, axis=1))
    if lambda_s == 0:
        return data_term
    smooth = 0.5 * lambda_s * jnp.dot(rho_flat, lap_fn(rho_flat))
    return data_term + smooth

# value‑and‑grad handle (J, ∇J)
loss_and_grad = jax.jit(jax.value_and_grad(lsq_loss))

###############################################################################
# Slice solver (quadratic ‑> CG)                                              #
###############################################################################

def cg_matvec(rho, idx, w, H, W, lambda_s):
    N = H*W
    n_obs = idx.shape[0]
    lap_fn = laplacian_2d(H, W)

    def single_TtT(x, i, wi):
        y = pic_forward(x, i, wi)
        return pic_adjoint(y, i, wi, N)

    A_rho = jax.vmap(single_TtT, in_axes=(None, 0, 0))(rho, idx, w).mean(0)
    if lambda_s:
        A_rho += lambda_s * lap_fn(rho)
    return A_rho


def pcg(Ax, b, x0, tol=1e-5, maxiter=100):
    r = b - Ax(x0)
    p = r
    rs_old = jnp.dot(r, r)

    def body(state):
        k, x, r, p, rs_old = state
        Ap = Ax(p)
        alpha = rs_old / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.dot(r_new, r_new)
        beta = rs_new / rs_old
        p_new = r_new + beta * p
        return k+1, x_new, r_new, p_new, rs_new

    def cond(state):
        k, _, r, _, rs = state
        return jnp.logical_and(k < maxiter, jnp.sqrt(rs) > tol)

    state0 = (0, x0, r, p, rs_old)
    _, x_final, _, _, _ = lax.while_loop(cond, body, state0)
    return x_final

###############################################################################
# solve_lsq_slice with optional gradient return                               #
###############################################################################

@partial(jax.jit, static_argnums=(4,5,6,7))
def solve_lsq_slice(rho_init,        # (N,)
                    y_obs,          # (n_obs, N)
                    idx, w,         # (n_obs,N,K)
                    H, W,
                    lambda_s=0.0,
                    return_grad=False):
    N = H*W
    Ax = lambda r: cg_matvec(r, idx, w, H, W, lambda_s)
    # RHS
    b = jax.vmap(lambda i, wi, y: pic_adjoint(y, i, wi, N),
                 in_axes=(0,0,0))(idx, w, y_obs).mean(0)
    rho_hat = pcg(Ax, b, rho_init)

    if not return_grad:
        return rho_hat
    loss, grad = loss_and_grad(rho_hat, y_obs, idx, w, H, W, lambda_s)
    return rho_hat, loss, grad

###############################################################################
# Demo / test                                                                 #
###############################################################################

if __name__ == "__main__":
    H, W = 64, 64
    m_src = m_tgt = (H, W)
    omega = jnp.array([0, W, 0, H], dtype=jnp.float32)
    xp    = jnp.stack(jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij"))
    xp    = xp.reshape(2, -1)

    idx, w = build_pic_stencils_2d(omega, m_src, m_tgt, xp)

    n_obs = 4
    key   = jax.random.PRNGKey(0)
    rho_gt= jax.random.normal(key, (H*W,))
    y     = jax.vmap(lambda i, wi: pic_forward(rho_gt, i, wi),
                     in_axes=(None,0,0))(idx.repeat(n_obs,0),
                                         w.repeat(n_obs,0)).reshape(n_obs, -1)

    rho0  = jnp.zeros_like(rho_gt)
    rho_est, loss, g = solve_lsq_slice(rho0, y, idx[None,...].repeat(n_obs,0),
                                       w[None,...].repeat(n_obs,0),
                                       H, W, 0.001, return_grad=True)
    print("loss", loss, "‖grad‖", jnp.linalg.norm(g))
"""
