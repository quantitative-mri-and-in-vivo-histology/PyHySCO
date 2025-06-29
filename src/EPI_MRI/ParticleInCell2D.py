import torch




def get_push_forward_matrix_2d_analytic(omega,
                                        m_source,
                                        m_target,
                                        xp,
                                        device=None,
                                        do_derivative=False,
                                        return_jacobian=False):
    """
    Construct a dense 2D push-forward matrix using analytic separable 1D basis functions.
    Optionally compute derivatives and Jacobian.

    Parameters
    ----------
    omega : list or torch.Tensor, shape (4,)
        Domain bounds [x0, x1, y0, y1].
    m_source : tuple (H, W)
        Input grid size (height and width).
    m_target : tuple (H, W)
        Output grid size (height and width).
    xp : torch.Tensor, shape (2, N)
        Particle positions in physical space.
    device : torch.device
        Device to use (optional).
    do_derivative : bool
        If True, return a derivative function dT(rho).
    return_jacobian : bool
        If True, also return the Jacobian vector Jac = T.sum(dim=1)

    Returns
    -------
    T : torch.Tensor, shape (H * W, N)
        Push-forward matrix mapping N particle weights to H*W grid.
    dT (optional) : function
        Callable derivative operator: dT(rho) → shape (H*W, N, 2)
    Jac (optional) : torch.Tensor
        Vector of length H*W giving the Jacobian mass scaling per voxel.
    """
    if device is None:
        device = xp.device

    # source grid settings
    n_particles = torch.prod(m_source)
    particle_size = (omega[1::2]-omega[0::2]) / m_source

    # target grid settings
    n_cells = torch.prod(m_target)
    cell_size = (omega[1::2]-omega[0::2]) / m_target

    x_vox = (xp[0] - omega[0]) / cell_size[0]
    y_vox = (xp[1] - omega[2]) / cell_size[1]

    x_vox = x_vox.permute(1,0)
    y_vox = y_vox.permute(1,0)

    Px = torch.floor(x_vox).long()
    wx = x_vox - Px.float()
    Py = torch.floor(y_vox).long()
    wy = y_vox - Py.float()
    Px = Px.reshape(-1)
    Py = Py.reshape(-1)
    wx = wx.reshape(-1)
    wy = wy.reshape(-1)

    # wx = x_vox - Px.float() - 0.5
    # wy = y_vox - Py.float() - 0.5

    pwidth = torch.ceil(particle_size / cell_size).to(torch.int32)

    # # # # Evaluate 1D basis
    Bx, Dx = int1DSingle(wx, pwidth[0], particle_size[0], cell_size[0], particle_size[0], do_derivative=do_derivative)  # [2*p+1, N]
    By, Dy = int1DSingle(wy, pwidth[1], particle_size[1], cell_size[1], particle_size[1], do_derivative=do_derivative)

    # Bx, Dx = int1DSingle(wx, pwidth[0], cell_size[0], particle_size[0], cell_size[0], do_derivative=do_derivative)  # [2*p+1, N]
    # By, Dy = int1DSingle(wy, pwidth[1], cell_size[1], particle_size[1], cell_size[1], do_derivative=do_derivative)

    Bx = Bx/Bx.sum(0)
    By = By/By.sum(0)

    nbx = Bx.shape[0]
    nby = By.shape[0]
    nVoxel = nbx * nby

    I = torch.empty(nVoxel * n_particles, dtype=torch.long, device=device)
    J = torch.empty(nVoxel * n_particles, dtype=torch.long, device=device)
    B = torch.empty(nVoxel * n_particles, dtype=xp.dtype, device=device)
    if do_derivative:
        dBx = torch.empty_like(B)
        dBy = torch.empty_like(B)

    pp = 0
    for i, px in enumerate(range(-pwidth[0], pwidth[0] + 1)):
        for j, py in enumerate(range(-pwidth[1], pwidth[1] + 1)):
            idx = slice(pp * n_particles, (pp + 1) * n_particles)
            pp += 1

            x_idx = Px + px
            y_idx = Py + py
            Iij = x_idx * m_target[0] + y_idx  # Flattened linear index

            Bij = Bx[i, :] * By[j, :]  # Elementwise per-particle weight

            I[idx] = Iij
            J[idx] = torch.arange(n_particles, device=Px.device)
            B[idx] = Bij

            if do_derivative:
                dBx[idx] = Dx[i, :] * By[j, :]
                dBy[idx] = Bx[i, :] * Dy[j, :]

    valid = (I >= 0) & (I < n_cells)
    I = I[valid]
    J = J[valid]
    B = B[valid]

    T = torch.sparse_coo_tensor(
        torch.stack((J, I)), B, size=(n_particles, n_cells)).coalesce()

    results = [T]

    if do_derivative:
        dBx = dBx[valid]
        dBy = dBy[valid]

        def dT(rho):
            return torch.stack([
                torch.sparse_coo_tensor(torch.stack((J, I)), rho[I] * dBx,
                                        size=(n_particles, n_cells)).coalesce(),
                torch.sparse_coo_tensor(torch.stack((J, I)), rho[I] * dBy,
                                        size=(n_particles, n_cells)).coalesce()
            ], dim=-1)

        results.append(dT)

    if return_jacobian:
        Jac = torch.zeros(n_particles, device=device)
        Jac = Jac.scatter_add(0, J, B)
        results.append(Jac)

    return tuple(results) if len(results) > 1 else results[0]

def int1DSingle(w, pwidth, eps, h, hp, do_derivative=False):
    """
    One-dimensional interpolation and optional derivative for distortion correction.

    Parameters
    ----------
    w : torch.Tensor
        Relative positions of particles within their cells.
    pwidth : int
        Support width of the basis function.
    eps : float
        Particle width.
    h : float
        Grid spacing.
    hp : float
        Cell size of the particle mesh.
    do_derivative : bool
        Whether to return the derivative (default: False).

    Returns
    -------
    Bij : torch.Tensor
        Interpolation weights of shape (2*pwidth+1, N).
    dBij : torch.Tensor (optional)
        Derivative of weights w.r.t. w, same shape as Bij.
    """
    N = w.shape[0]
    Bij = torch.zeros((2 * pwidth + 1, N), dtype=w.dtype, device=w.device)
    if do_derivative:
        dBij = torch.zeros_like(Bij)

    # Initial B and b values (left edge of first bin)
    Bleft, bleft = B_single(-pwidth - w, eps, h,
                            do_derivative=True) if do_derivative \
        else (B_single(-pwidth - w, eps, h), None)

    for p in range(-pwidth, pwidth + 1):
        idx = p + pwidth
        Bright, bright = B_single(p + 1 - w, eps, h,
                                  do_derivative=True) if do_derivative \
            else (B_single(p + 1 - w, eps, h), None)

        Bij[idx, :] = hp * (Bright - Bleft).squeeze()
        if do_derivative:
            dBij[idx, :] = -hp * (bright - bleft).squeeze()

        Bleft = Bright
        if do_derivative:
            bleft = bright

    return (Bij, dBij) if do_derivative else Bij



def B_single(x, eps, h, do_derivative=False):
    """
    Compute 1D PIC interpolation weights (and optionally their derivatives).

    Parameters
    ----------
    x : torch.Tensor
        Input data (e.g. offset from grid point).
    eps : float
        Particle width.
    h : float
        Grid cell size in the interpolation dimension.
    do_derivative : bool, optional
        Whether to return the derivative w.r.t. x.

    Returns
    -------
    Bij : torch.Tensor
        Interpolation weight.
    dBij : torch.Tensor (optional)
        Derivative of interpolation weight w.r.t. x.
    """
    Bij = torch.zeros_like(x)
    if do_derivative:
        dBij = torch.zeros_like(x)

    thresh = eps / h

    ind1 = (-thresh <= x) & (x <= 0)
    ind2 = (0 < x) & (x <= thresh)
    ind3 = (x > thresh)

    # Weight
    Bij[ind1] = x[ind1] + 0.5 * (h / eps) * x[ind1] ** 2 + eps / (2 * h)
    Bij[ind2] = x[ind2] - 0.5 * (h / eps) * x[ind2] ** 2 + eps / (2 * h)
    Bij[ind3] = eps / h
    Bij = Bij / eps

    if do_derivative:
        dBij[ind1] = 1 + h * x[ind1] / eps
        dBij[ind2] = 1 - h * x[ind2] / eps
        dBij[ind1 | ind2] /= eps  # outside these, dBij is 0 by default

        return Bij, dBij
    else:
        return Bij





#import torch
#
#
# def get_push_forward_matrix_2d_analytic(omega, mc, xp, h, hp,
#                                         device=None, do_derivative=False,
#                                         return_jacobian=False):
#     """
#     Construct a dense 2D push-forward matrix using analytic separable 1D basis functions.
#     Optionally compute derivatives and Jacobian.
#
#     Parameters
#     ----------
#     omega : list or torch.Tensor, shape (4,)
#         Domain bounds [x0, x1, y0, y1].
#     mc : tuple (H, W)
#         Output grid size (height and width).
#     xp : torch.Tensor, shape (2, N)
#         Particle positions in physical space.
#     h : tuple (hx, hy)
#         Output voxel sizes.
#     hp : tuple (hpx, hpy)
#         Particle cell sizes.
#     device : torch.device
#         Device to use (optional).
#     do_derivative : bool
#         If True, return a derivative function dT(rho).
#     return_jacobian : bool
#         If True, also return the Jacobian vector Jac = T.sum(dim=1)
#
#     Returns
#     -------
#     T : torch.Tensor, shape (H * W, N)
#         Push-forward matrix mapping N particle weights to H*W grid.
#     dT (optional) : function
#         Callable derivative operator: dT(rho) → shape (H*W, N, 2)
#     Jac (optional) : torch.Tensor
#         Vector of length H*W giving the Jacobian mass scaling per voxel.
#     """
#     if device is None:
#         device = xp.device
#
#     dtype = xp.dtype
#     H, W = mc
#     N = H*W # number of particles
#     total_voxels = H * W
#
#     h = torch.tensor(h, device=device)
#     hp = torch.tensor(hp, device=device)
#     epsP = hp
#     pwidth = torch.ceil(epsP / h).to(torch.int32)
#
#     x_vox = (xp[0] - omega[0]) / h[0]
#     y_vox = (xp[1] - omega[2]) / h[1]
#
#     Px = torch.floor(x_vox).long()
#     wx = x_vox - Px.float()
#     Py = torch.floor(y_vox).long()
#     wy = y_vox - Py.float()
#
#     Px = Px.view(-1)
#     Py = Py.view(-1)
#     wx = wx.view(-1)
#     wy = wy.view(-1)
#
#     # Evaluate 1D basis
#     Bx, Dx = int1DSingle(wx, pwidth[0], epsP[0], h[0], hp[0], do_derivative=do_derivative)  # [2*p+1, N]
#     By, Dy = int1DSingle(wy, pwidth[1], epsP[1], h[1], hp[1], do_derivative=do_derivative)
#
#     nbx = Bx.shape[0]
#     nby = By.shape[0]
#     nVoxel = nbx * nby
#
#     I = torch.empty(nVoxel * N, dtype=torch.long, device=device)
#     J = torch.empty(nVoxel * N, dtype=torch.long, device=device)
#     B = torch.empty(nVoxel * N, dtype=dtype, device=device)
#     if do_derivative:
#         dBx = torch.empty_like(B)
#         dBy = torch.empty_like(B)
#
#     pp = 0
#     for i, px in enumerate(range(-pwidth[0], pwidth[0] + 1)):
#         for j, py in enumerate(range(-pwidth[1], pwidth[1] + 1)):
#             idx = slice(pp * N, (pp + 1) * N)
#             pp += 1
#
#             x_idx = Px + px
#             y_idx = Py + py
#             Iij = x_idx * W + y_idx  # Flattened linear index
#
#             Bij = Bx[i, :] * By[j, :]  # Elementwise per-particle weight
#
#             I[idx] = Iij
#             J[idx] = torch.arange(N, device=Px.device)
#             B[idx] = Bij
#
#             if do_derivative:
#                 dBx[idx] = Dx[i, :] * By[j, :]
#                 dBy[idx] = Bx[i, :] * Dy[j, :]
#
#     valid = (I >= 0) & (I < total_voxels)
#     I = I[valid]
#     J = J[valid]
#     B = B[valid]
#
#     T = torch.sparse_coo_tensor(
#         torch.stack((I, J)), B, size=(H * W, N)).coalesce()
#
#     results = [T]
#
#     if do_derivative:
#         dBx = dBx[valid]
#         dBy = dBy[valid]
#
#         def dT(rho):
#             return torch.stack([
#                 torch.sparse_coo_tensor(torch.stack((I, J)), rho[J] * dBx,
#                                         size=(H * W, N)).coalesce(),
#                 torch.sparse_coo_tensor(torch.stack((I, J)), rho[J] * dBy,
#                                         size=(H * W, N)).coalesce()
#             ], dim=-1)
#
#         results.append(dT)
#
#     if return_jacobian:
#         Jac = torch.zeros(H * W, dtype=dtype, device=device)
#         Jac.index_add_(0, I, B)
#         results.append(Jac)
#
#     return tuple(results) if len(results) > 1 else results[0]
#
# def int1DSingle(w, pwidth, eps, h, hp, do_derivative=False):
#     """
#     One-dimensional interpolation and optional derivative for distortion correction.
#
#     Parameters
#     ----------
#     w : torch.Tensor
#         Relative positions of particles within their cells.
#     pwidth : int
#         Support width of the basis function.
#     eps : float
#         Particle width.
#     h : float
#         Grid spacing.
#     hp : float
#         Cell size of the particle mesh.
#     do_derivative : bool
#         Whether to return the derivative (default: False).
#
#     Returns
#     -------
#     Bij : torch.Tensor
#         Interpolation weights of shape (2*pwidth+1, N).
#     dBij : torch.Tensor (optional)
#         Derivative of weights w.r.t. w, same shape as Bij.
#     """
#     N = w.shape[0]
#     Bij = torch.zeros((2 * pwidth + 1, N), dtype=w.dtype,
#                       device=w.device)
#     if do_derivative:
#         dBij = torch.zeros_like(Bij)
#
#     # Initial B and b values
#     Bleft, bleft = B_single(-pwidth - w, eps, h,
#                                  do_derivative=True) if do_derivative \
#         else (B_single(-pwidth - w, eps, h), None)
#
#     for p in range(-pwidth, pwidth + 1):
#         idx = p + pwidth
#         Bright, bright = B_single(1 + p - w, eps, h,
#                                        do_derivative=True) if do_derivative \
#             else (B_single(1 + p - w, eps, h), None)
#
#         Bij[idx, :] = hp * (Bright - Bleft).squeeze()
#         if do_derivative:
#             dBij[idx, :] = -hp * (bright - bleft).squeeze()
#
#         Bleft = Bright
#         if do_derivative:
#             bleft = bright
#
#     return (Bij, dBij) if do_derivative else Bij
#
# def B_single(x, eps, h, do_derivative=False):
#     """
#     Compute 1D PIC interpolation weights (and optionally their derivatives).
#
#     Parameters
#     ----------
#     x : torch.Tensor
#         Input data (e.g. offset from grid point).
#     eps : float
#         Particle width.
#     h : float
#         Grid cell size in the interpolation dimension.
#     do_derivative : bool, optional
#         Whether to return the derivative w.r.t. x.
#
#     Returns
#     -------
#     Bij : torch.Tensor
#         Interpolation weight.
#     dBij : torch.Tensor (optional)
#         Derivative of interpolation weight w.r.t. x.
#     """
#     Bij = torch.zeros_like(x)
#     if do_derivative:
#         dBij = torch.zeros_like(x)
#
#     thresh = eps / h
#
#     ind1 = (-thresh <= x) & (x <= 0)
#     ind2 = (0 < x) & (x <= thresh)
#     ind3 = (x > thresh)
#
#     # Weight
#     Bij[ind1] = x[ind1] + 0.5 * (h / eps) * x[ind1] ** 2 + eps / (2 * h)
#     Bij[ind2] = x[ind2] - 0.5 * (h / eps) * x[ind2] ** 2 + eps / (2 * h)
#     Bij[ind3] = eps / h
#     Bij = Bij / eps
#
#     if do_derivative:
#         dBij[ind1] = 1 + h * x[ind1] / eps
#         dBij[ind2] = 1 - h * x[ind2] / eps
#         dBij[ind1 | ind2] /= eps  # outside these, dBij is 0 by default
#
#         return Bij, dBij
#     else:
#         return Bij


def drop_rows_from_sparse_matrix(T, valid_rows):
    """
    Drop rows from a sparse COO matrix T using a boolean row mask.

    Parameters
    ----------
    T : torch.sparse_coo_tensor
        The sparse matrix of shape [N_rows, N_cols].
    valid_rows : torch.BoolTensor
        Boolean mask of shape [N_rows] indicating which rows to keep.

    Returns
    -------
    filtered_T : torch.sparse_coo_tensor
        New sparse matrix with only valid rows.
    valid_row_indices : torch.LongTensor
        Indices of kept rows (useful for mapping).
    """
    C = T.coalesce()

    # 1. Get row indices of non-zero entries
    row_idx = C.indices()[0]
    col_idx = C.indices()[1]
    values = C.values()

    # 2. Keep only entries whose row is marked as valid
    keep_mask = valid_rows[row_idx]
    row_idx_new_raw = row_idx[keep_mask]
    col_idx_new = col_idx[keep_mask]
    values_new = values[keep_mask]

    # 3. Remap row indices to new positions
    valid_row_indices = torch.nonzero(valid_rows).squeeze(1)
    old_to_new = -torch.ones_like(valid_rows, dtype=torch.long)
    old_to_new[valid_row_indices] = torch.arange(len(valid_row_indices),
                                                 device=T.device)
    row_idx_new = old_to_new[row_idx_new_raw]

    # 4. Reassemble sparse matrix
    new_indices = torch.stack([row_idx_new, col_idx_new], dim=0)
    filtered_T = torch.sparse_coo_tensor(
        new_indices,
        values_new,
        size=(len(valid_row_indices), T.shape[1]),
        dtype=T.dtype,
        device=T.device
    )

    return filtered_T, valid_row_indices