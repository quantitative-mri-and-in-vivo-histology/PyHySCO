from __future__ import annotations

from functools import partial
import jax.numpy as jnp
import jax


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


################################################################################
# 1 ─── 1‑D integrated B‑spline kernel                                        #
################################################################################

@partial(jax.jit, static_argnames=("pwidth",))
def _b_kernel_integrated(frac: jnp.ndarray,
                         eps:  float,
                         h:    float,
                         pwidth: int = 1) -> jnp.ndarray:
    """Return (*2·pwidth+1*, N) array of 1‑D antiderivative weights.

    * `frac`   – fractional offset in [0,1) for each particle (shape N)
    * `eps/h`  – support ratio (normally 1 for quadratic B‑spline)
    * pwidth   – half‑width of support in cells (usually 1)
    """
    def single_offset(p):
        x_left  =  p - pwidth   - frac              # left edge of bin
        x_right = x_left + 1.0                      # right edge
        Bij = _B_single(x_right, eps, h) - _B_single(x_left, eps, h)
        return Bij * h              # scale from int to weight

    out = jax.vmap(single_offset)(jnp.arange(-pwidth, pwidth + 1))  # (2p+1, N)
    return out

@partial(jax.jit, static_argnames=())
def _B_single(x: jnp.ndarray, eps: float, h: float) -> jnp.ndarray:
    """Scalar antiderivative of quadratic B‑spline (C¹)."""
    t = eps / h
    a = x + 0.5 * (h/eps) * x**2 + eps/(2*h)
    b = x - 0.5 * (h/eps) * x**2 + eps/(2*h)
    c = jnp.full_like(x, eps/h)
    y = jnp.where(x <= 0, a, b)
    y = jnp.where(x > t, c, y)
    y = jnp.where(x < -t, 0.0, y)
    return y / eps

################################################################################
# 2 ─── Factory: make_stencil_builder                                         #
################################################################################

def make_stencil_builder(omega: jnp.ndarray,
                         m_cell:  Tuple[int, int, int],
                         m_part:  Tuple[int, int, int],
                         pwidth: int = 1) -> callable:
    """Return a **jit‑compiled** function  (xp → (indices, weights))."""

    Dz, Dy, Dx = m_cell
    extent     = omega[1::2] - omega[::2]
    cell_sz    = extent / jnp.array(m_cell)
    part_sz    = extent / jnp.array(m_part)
    eps        = part_sz

    # Pre‑compute offset list once (Python constant → fused)
    offsets: List[Tuple[int,int,int]] = [
        (dz, dy, dx)
        for dz in range(-pwidth, pwidth + 1)
        for dy in range(-pwidth, pwidth + 1)
        for dx in range(-pwidth, pwidth + 1)
    ]
    offsets = jnp.array(offsets, dtype=jnp.int32)           # (S, 3)
    S       = offsets.shape[0]

    @partial(jax.jit, static_argnames=())
    def stencil_fn(xp: jnp.ndarray):                        # xp (3, N)
        # 1. Voxel coordinates & fractional parts --------------------------------
        # xv = (xp - omega[::2, None]) / part_sz[:, None]      # (3, N)
        # base = jnp.floor(xv).astype(jnp.int32)               # (3, N)
        # frac = xv - base                                     # (3, N)

        xv = (xp - omega[::2, None]) / cell_sz[:, None]  # (3, N)
        base = jnp.floor(xv).astype(jnp.int32)  # (3, N)
        frac = xv - base

        # 2. 1‑D weights per axis ------------------------------------------------------------------
        Bz = _b_kernel_integrated(frac[0], eps[0], cell_sz[0], pwidth)  # (2p+1, N)
        By = _b_kernel_integrated(frac[1], eps[1], cell_sz[1], pwidth)
        Bx = _b_kernel_integrated(frac[2], eps[2], cell_sz[2], pwidth)

        # 3. Combine axes -------------------------------------------------------------------------
        def combine(offset):
            dz, dy, dx = offset
            w = (Bz[dz + pwidth] * By[dy + pwidth] * Bx[dx + pwidth])    # (N,)
            idx3 = base + jnp.array([[dz], [dy], [dx]], dtype=jnp.int32)  # (3,N)
            lin  = (idx3[0] * Dy + idx3[1]) * Dx + idx3[2]               # (N,)
            return lin, w

        lin_idx, wts = jax.vmap(combine)(offsets)  # each (S, N)
        indices = lin_idx.T                        # (N, S)
        weights = wts.T                            # (N, S)
        return indices, weights

    return stencil_fn
