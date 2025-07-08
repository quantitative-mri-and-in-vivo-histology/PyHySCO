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



#
#
#
# def make_stencil_builder(omega, m_source, m_target):
#     Dz, Dy, Dx = m_source
#     Pz, Py, Px = m_target
#
#     extent = omega[1::2] - omega[::2]
#     cell_sz = extent / jnp.array(m_source)
#     part_sz = extent / jnp.array(m_target)
#     eps = part_sz
#     pwidth = 1
#
#     @partial(jax.jit, static_argnums=())
#     def stencil_fn(xp: jnp.ndarray):
#         # xp: (3, N)
#         z_vox = (xp[0] - omega[0]) / part_sz[0]
#         y_vox = (xp[1] - omega[2]) / part_sz[1]
#         x_vox = (xp[2] - omega[4]) / part_sz[2]
#
#         Pz_idx = jnp.floor(z_vox).astype(int)
#         Py_idx = jnp.floor(y_vox).astype(int)
#         Px_idx = jnp.floor(x_vox).astype(int)
#
#         wz = z_vox - jnp.floor(z_vox)
#         wy = y_vox - jnp.floor(y_vox)
#         wx = x_vox - jnp.floor(x_vox)
#
#         Bz = int1DSingle(wz, pwidth, eps[0], cell_sz[0], part_sz[0])
#         By = int1DSingle(wy, pwidth, eps[1], cell_sz[1], part_sz[1])
#         Bx = int1DSingle(wx, pwidth, eps[2], cell_sz[2], part_sz[2])
#
#         stencil_sz = (2 * pwidth + 1) ** 3
#         N = xp.shape[1]
#
#         Ilist = []
#         Blist = []
#         for iz in range(-pwidth, pwidth + 1):
#             for iy in range(-pwidth, pwidth + 1):
#                 for ix in range(-pwidth, pwidth + 1):
#                     z_idx = Pz_idx + iz
#                     y_idx = Py_idx + iy
#                     x_idx = Px_idx + ix
#                     Iijk = (z_idx * Dy + y_idx) * Dx + x_idx
#                     Bij = Bz[iz + pwidth] * By[iy + pwidth] * Bx[ix + pwidth]
#                     Ilist.append(Iijk)
#                     Blist.append(Bij)
#
#         I = jnp.stack(Ilist, axis=0).T  # (N, stencil_sz)
#         B = jnp.stack(Blist, axis=0).T  # (N, stencil_sz)
#         return I, B
#
#     return stencil_fn
#
#
#
# @partial(jax.jit, static_argnums=(1,2,4))
# def build_pic_stencils_3d(
#         omega,              # (z0,z1,y0,y1,x0,x1)
#         m_source,           # (Dz, Dy, Dx)
#         m_target,           # (Pz, Py, Px)
#         xp,                 # (3, Np)  order: (z,y,x)
#         return_jacobian=False
#     ):
#     Dz, Dy, Dx = m_source
#     Pz, Py, Px = m_target
#     n_cells     = Dz * Dy * Dx
#     n_particles = Pz * Py * Px
#
#     extent      = omega[1::2] - omega[0::2]          # (Lz,Ly,Lx)
#     cell_sz     = extent / jnp.array(m_source)       # Δz,Δy,Δx
#     part_sz     = extent / jnp.array(m_target)       # δz,δy,δx
#
#     # particle positions in voxel space
#     z_vox = (xp[0] - omega[0]) / part_sz[0]
#     y_vox = (xp[1] - omega[2]) / part_sz[1]
#     x_vox = (xp[2] - omega[4]) / part_sz[2]
#
#     Pz_idx, wz = jnp.floor(z_vox).astype(int), z_vox - jnp.floor(z_vox)
#     Py_idx, wy = jnp.floor(y_vox).astype(int), y_vox - jnp.floor(y_vox)
#     Px_idx, wx = jnp.floor(x_vox).astype(int), x_vox - jnp.floor(x_vox)
#
#     Pz_idx, Py_idx, Px_idx, wz, wy, wx = map(
#         lambda a: a.reshape(-1),
#         (Pz_idx, Py_idx, Px_idx, wz, wy, wx)
#     )
#
#     epsP   = 1 * part_sz
#     # pwidth = jnp.array(jnp.ceil(epsP / cell_sz)).astype(int)     # (pz,py,px)
#     pwidth = jnp.array([1, 1, 1], dtype=jnp.int32)
#
#     Bz = int1DSingle(wz, 1, epsP[0], cell_sz[0], part_sz[0])
#     By = int1DSingle(wy, 1, epsP[1], cell_sz[1], part_sz[1])
#     Bx = int1DSingle(wx, 1, epsP[2], cell_sz[2], part_sz[2])
#
#     nz, ny, nx = map(lambda b: b.shape[0], (Bz, By, Bx))
#     stencil_sz = nz * ny * nx
#
#     PWIDTH_Z = 1
#     PWIDTH_Y = 1
#     PWIDTH_X = 1
#     Ilist, Jlist, Blist = [], [], []
#     pp = 0
#     for iz, dz in enumerate(range(-PWIDTH_Z, PWIDTH_Z + 1)):
#         for iy, dy in enumerate(range(-PWIDTH_Y, PWIDTH_Y + 1)):
#             for ix, dx in enumerate(range(-PWIDTH_X, PWIDTH_X + 1)):
#
#                 # neighbour voxel indices
#                 z_idx = Pz_idx + dz
#                 y_idx = Py_idx + dy
#                 x_idx = Px_idx + dx
#
#                 # linear index
#                 Iijk = (z_idx * Dy + y_idx) * Dx + x_idx
#
#                 Bij  = Bz[iz] * By[iy] * Bx[ix]
#
#                 Ilist.append(Iijk)
#                 Jlist.append(jnp.arange(n_cells))
#                 Blist.append(Bij)
#                 pp += 1
#
#     I = jnp.stack(Ilist, 0).reshape(stencil_sz, n_particles).T
#     B = jnp.stack(Blist, 0).reshape(stencil_sz, n_particles).T
#
#     if return_jacobian:
#         Jac = jnp.zeros(n_cells, B.dtype).at[
#             jnp.stack(Jlist,0).reshape(-1)
#         ].add(jnp.stack(Blist,0).reshape(-1))
#         return I, B, Jac
#     return I, B
#
# # @partial(jax.jit, static_argnums=(1,2))
# def B_single(x: jnp.ndarray, eps: float, h: float):
#     """
#     Scalar antiderivative of the PIC kernel (JIT-friendly version).
#     Works for arbitrary traced `x` (no boolean indexing).
#     """
#     thresh = eps / h
#
#     # region 1 : -thresh <= x <= 0
#     a = x + 0.5 * (h / eps) * x**2 + eps / (2 * h)
#
#     # region 2 : 0 < x <=  thresh
#     b = x - 0.5 * (h / eps) * x**2 + eps / (2 * h)
#
#     # region 3 :            x >  thresh
#     c = jnp.full_like(x, eps / h)
#
#     # piece-wise assembly with nested where’s
#     Bij = jnp.where(x <= 0, a, b)           # choose a or b
#     Bij = jnp.where(x > thresh, c, Bij)     # overwrite by c if region 3
#     Bij = jnp.where(x < -thresh, 0.0, Bij)  # zero outside support
#
#     return Bij / eps
#
#
# @partial(jax.jit, static_argnums=(1))
# def int1DSingle(w: jnp.ndarray,
#                     pwidth: int,
#                     eps:    float,
#                     h:      float,
#                     hp:     float):
#     """
#     jnp implementation of `int1DSingle` (without derivatives).
#     Returns Bij  with shape  (2*pwidth+1, N)
#     """
#     # N   = w.shape[0]
#     N = 108900
#     Bij = jnp.zeros((2 * pwidth + 1, N), dtype=w.dtype)
#
#     # Bleft at left edge of first bin
#     Bleft = B_single(-pwidth - w, eps, h)
#     for p in range(-pwidth, pwidth + 1):
#         idx     = p + pwidth
#         Bright  = B_single(p + 1 - w, eps, h)
#         Bij     = Bij.at[idx].set(hp * (Bright - Bleft))
#         Bleft   = Bright
#     return Bij