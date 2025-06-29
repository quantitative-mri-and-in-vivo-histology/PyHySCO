import math
from EPI_MRI.utils import *
import torchsparsegradutils as tsgu
from EPI_MRI.LinearOperators import *


class LeastSquaresCorrectionMultiPeSparse4d:
    """
    Given a field map, produces a corrected image using multiple PE-RPE pairs.

    This class provides functionality to correct distorted images using a field map.
    The method is based on Least Squares Restoration, combining information from
    multiple phase encoding directions.

    Reference: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ApplytopupFurtherInformation

    Attributes
    ----------
    dataObj : `MultiPeDataObject`
        Contains original (distorted) data with multiple PE-RPE pairs.
    A : `LinearOperators.LinearOperator`
        Averaging operator used for image averaging, e.g. `LinearOperators.myAvg1D`.
    device : str
        The device on which to compute operations, e.g., 'cpu' or 'cuda'.
    xc : torch.Tensor (size m)
        Cell-centered grid in the phase encoding dimension.

    Parameters
    ----------
    data : `MultiPeDataObject`
        Contains the original (distorted) data with multiple PE-RPE pairs.
    A : `LinearOperators.LinearOperator`
        The averaging operator to be used for image averaging, e.g. `LinearOperators.myAvg1D`.
    """
    def __init__(self, data, A):
        self.dataObj = data
        self.A = A
        self.device = data.device
        self.xc = get_cell_centered_grid(self.dataObj.omega, self.dataObj.m, device=self.device, dtype=self.dataObj.dtype)

    def apply_correction(self, yc):
        """
        Applies the distortion correction to the input data.

        Parameters
        ----------
        yc : torch.Tensor
            The input data.

        Returns
        ----------
        rhocorr : torch.Tensor (size m)
            The corrected image.
        """
        # Initialize lists to store matrices and data
        C_list = []
        rho_list = []
        alpha = 1e-3
        thres = 1e-3

        # --- NEW: Get full coordinate grid (all dims) ---
        xc = get_cell_centered_grid(self.dataObj.omega[2:], self.dataObj.m[1:],
                                    device=self.device,
                                    dtype=self.dataObj.dtype,
                                    return_all=True)

        # Process each PE-RPE pair
        for i, pair in enumerate(self.dataObj.image_pairs):

            v = torch.tensor([0.0, 0.0, 1.0], device=self.device,
                             dtype=self.dataObj.dtype)
            bc1 = self.A.mat_mul(yc).reshape(-1, 1)  # averaging matrix & translation vector
            bc1_full = bc1 * v.view(1, -1)  # shift vector in original space
            xp1 = xc.view(3,-1) + bc1_full.T
            xp2 = xc.view(3,-1) - bc1_full.T

            P = \
            torch.eye(3, device=self.dataObj.device, dtype=self.dataObj.dtype)[
                self.dataObj.permute[i][1:]]
            T = torch.tensor(self.dataObj.rel_mats[i][:3, :3],
                             device=self.dataObj.device,
                             dtype=self.dataObj.dtype)
            T_permuted = P @ T @ P.T
            v_rot = T_permuted @ v
            # v_rot = torch.round(v_rot)

            center = 0.5 * (
                        torch.tensor(self.dataObj.omega[3::2]) + torch.tensor(
                    self.dataObj.omega[2::2]))  # (x_c, y_c, z_c)

            xp1 = T_permuted @ (xp1 - center.unsqueeze(1)) + center.view(3, 1)
            xp2 = T_permuted @ (xp2 - center.unsqueeze(1)) + center.view(3, 1)

            xp1 = xp1[1:, :].reshape(2, *self.dataObj.m[1:])
            xp2 = xp2[1:, :].reshape(2, *self.dataObj.m[1:])

            xp1 = xp1.reshape(2, *self.dataObj.m[1:])
            xp2 = xp2.reshape(2, *self.dataObj.m[1:])

            x0, x1, y0, y1 = self.dataObj.omega[-4:]

            x1_coords = xp1[0, i, :, :]
            y1_coords = xp1[1, i, :, :]
            x2_coords = xp2[0, i, :, :]
            y2_coords = xp2[1, i, :, :]

            # Define mask: True where both xp1 and xp2 are valid
            valid_mask = (
                    (x1_coords >= x0) & (x1_coords <= x1) &
                    (y1_coords >= y0) & (y1_coords <= y1) &
                    (x2_coords >= x0) & (x2_coords <= x1) &
                    (y2_coords >= y0) & (y2_coords <= y1)
            )  # shape (H, W)

            # Flatten it to 1D
            valid_mask_flat = valid_mask.flatten()  # shape (H*W,)

            C1_slices = []
            C2_slices = []
            for slice_index in range(self.dataObj.m[1]):
                C1_slice = self.get_push_forward_matrix_2d_analytic(
                    self.dataObj.omega[-4:],
                    self.dataObj.m[-2:],
                    xp1[:,slice_index,:,:].clone(),
                    self.dataObj.h[-2:],
                    self.dataObj.h[-2:]
                )

                # Apply to C1_slice, C2_slice and corresponding rho rows
                # C1_slice = C1_slice[valid_mask_flat, :]
                # C1_slice = self.drop_pushforward_matrix_rows(C1_slice, thres)
                C1_slices.append(C1_slice)

                C2_slice = self.get_push_forward_matrix_2d_analytic(
                    self.dataObj.omega[-4:],
                    self.dataObj.m[-2:],
                    xp2[:, slice_index, :, :].clone(),
                    self.dataObj.h[-2:],
                    self.dataObj.h[-2:]
                )
                # C2_slice = C2_slice[valid_mask_flat, :]
                # C2_slice = self.drop_pushforward_matrix_rows(C2_slice, thres)
                C2_slices.append(C2_slice)

            C1 = torch.stack(C1_slices, dim=0)
            C2 = torch.stack(C2_slices, dim=0)

            C = torch.cat((C1, C2), dim=1)

            # Store matrices and data
            C_list.append(C)



            rho0 = pair.pe_image
            rho1 = pair.rpe_image

            rho0 = rho0.reshape(self.dataObj.m[0], self.dataObj.m[1], -1)
            rho1 = rho1.reshape(self.dataObj.m[0], self.dataObj.m[1], -1)

            rho_list.append(torch.cat((rho0, rho1), dim=-1))

        # Concatenate all matrices and data
        C_all = torch.cat(C_list, dim=1)
        rho_all = torch.cat(rho_list, dim=-1)


        rhocorr_vols = []
        for vol_index in range(self.dataObj.m[0]):
            rhocorr_slices = []
            for slice_index in range(self.dataObj.m[1]):
                A = C_all[slice_index]
                b = rho_all[vol_index,slice_index].unsqueeze(1)

                A, row_idx = self.drop_pushforward_matrix_rows(A, thres)
                b = b[row_idx]

                L = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
                L = L.to_sparse_coo()
                sqrt_lam = torch.sqrt(
                    torch.tensor(1, dtype=A.dtype, device=A.device))
                A_reg = torch.cat([A, sqrt_lam * L], dim=0)
                b_reg = torch.cat([b,
                                   torch.zeros((L.shape[0], 1), dtype=b.dtype,
                                               device=b.device)], dim=0)

                rhocorr = tsgu.sparse_lstsq.sparse_generic_lstsq(A_reg, b_reg)
                rhocorr_slices.append(rhocorr)
            rhocorr_vol = torch.cat(rhocorr_slices, dim=0)
            rhocorr_vols.append(rhocorr_vol)


        rhocorr = torch.cat(rhocorr_vols, dim=0)
        full_rhocorr = rhocorr.reshape(*self.dataObj.m)
        return full_rhocorr



    def get_push_forward_matrix_2d_analytic(self, omega, mc, xp, h, hp,
                                            device=None):
        """
        Construct a dense 2D push-forward matrix using analytic separable 1D basis functions.

        Parameters
        ----------
        omega : list or torch.Tensor, shape (4,)
            Domain bounds [x0, x1, y0, y1].
        mc : tuple (H, W)
            Output grid size (height and width).
        xp : torch.Tensor, shape (2, N)
            Particle positions in physical space.
        h : tuple (hx, hy)
            Output voxel sizes.
        hp : tuple (hpx, hpy)
            Particle cell sizes.
        int1D_func : callable
            Function implementing 1D analytic integration (same as FAIR's int1D).
        device : torch.device
            Device to use (optional).

        Returns
        -------
        T : torch.Tensor, shape (H * W, N)
            Push-forward matrix mapping N particle weights to H*W grid.
        """
        if device is None:
            device = xp.device

        dtype = xp.dtype
        H, W = mc
        N = H*W

        h = torch.tensor(h, device=device)
        hp = torch.tensor(hp, device=device)
        epsP = hp
        pwidth = torch.ceil(epsP / h).to(torch.int32)

        x_vox = (xp[0] - omega[0]) / h[0]
        y_vox = (xp[1] - omega[2]) / h[1]

        Px = torch.floor(x_vox)  # instead of ceil
        wx = x_vox - (Px).float()
        Py = torch.floor(y_vox)  # instead of ceil
        wy = y_vox - (Py).float()

        Px = Px.view(-1)
        Py = Py.view(-1)
        wx = wx.view(-1)
        wy = wy.view(-1)

        Bx = self.int1DSingle(wx, pwidth[0], epsP[0], h[0], hp[0])
        By = self.int1DSingle(wy, pwidth[1], epsP[1], h[1], hp[1])

        # Repeat particle indices for each offset
        nbx = Bx.shape[0]
        nby = By.shape[0]
        nVoxel = nbx * nby

        # Allocate big arrays (like in MATLAB)
        I = torch.empty(N * nVoxel, dtype=torch.long, device=Px.device)
        J = torch.empty(N * nVoxel, dtype=torch.long, device=Px.device)
        B = torch.empty(N * nVoxel, dtype=Bx.dtype, device=Px.device)

        pp = 0
        for i, px in enumerate(range(-pwidth[0], pwidth[0] + 1)):
            for j, py in enumerate(range(-pwidth[1], pwidth[1] + 1)):
                idx = slice(pp * N, (pp + 1) * N)
                pp += 1

                x_idx = Px + px
                y_idx = Py + py
                Iij = x_idx * W + y_idx  # Flattened linear index

                Bij =  Bx[i, :] * By[j, :]  # Elementwise per-particle weight

                I[idx] = Iij
                J[idx] = torch.arange(N, device=Px.device)
                B[idx] = Bij

        # Stack everything
        valid = (I >= 0) & (I < H * W)
        I = I[valid]
        J = J[valid]
        B = B[valid]

        n_rows = N
        n_cols = N
        indices = torch.stack([I, J], dim=0)  # shape (2, K)
        values = B  # shape (K,)

        T = torch.sparse_coo_tensor(indices, values,
                                           size=(n_rows, n_cols),
                                           device=self.device, dtype=B.dtype)

        return T


    def drop_pushforward_matrix_rows(self, T, thres):
        C = T.coalesce()

        # 2. Compute row sums
        row_sums = torch.zeros(C.size(0), device=C.device, dtype=C.dtype)
        row_sums.index_add_(0, C.indices()[0], C.values())

        # 3. Identify valid rows
        valid_rows = row_sums >= thres
        valid_row_indices = torch.nonzero(valid_rows).squeeze(1)

        # 4. Build mapping from old row indices to new ones
        old_to_new = -torch.ones(C.size(0), device=C.device, dtype=torch.long)
        old_to_new[valid_row_indices] = torch.arange(len(valid_row_indices),
                                                     device=C.device)

        # 5. Select only entries in valid rows
        keep_mask = valid_rows[C.indices()[0]]
        new_indices_raw = C.indices()[:, keep_mask]
        new_values = C.values()[keep_mask]

        # 6. Remap old row indices to new indices
        new_rows = old_to_new[new_indices_raw[0]]
        new_cols = new_indices_raw[1]
        new_indices = torch.stack([new_rows, new_cols], dim=0)

        # 7. Create reduced-size sparse matrix
        filtered_C = torch.sparse_coo_tensor(
            new_indices,
            new_values,
            size=(len(valid_row_indices), C.size(1)),
            # new row count, original column count
            dtype=C.dtype,
            device=C.device
        )

        return filtered_C, valid_rows


    def get_push_forward_parallel(self, omega, mc, mf, xp, h, hp, transform_mat=None):
        """
        Constructs the push forward matrix for distortion correction.

        Parameters
        ----------
        omega : torch.Tensor (size 2*dim)
            The image domain.
        mc : int
            The size of the distortion dimension.
        mf : int
            The size of the non-distortion dimensions.
        xp : torch.tensor (size (-1, m[-1]))
            The distorted grid in reference space.
        h : float
            The cell-size in the distortion dimension.
        hp : float
            The cell-size in the distortion dimension.
        transform_mat : torch.Tensor, optional
            Transformation matrix that maps from reference space to target space.

        Returns
        ----------
        T : torch.Tensor (size mf, mc, xp.shape[1])
            The push forward matrix that maps from reference space to target space.
        """
        epsP = hp  # width of particles
        n_parallel = mf
        np = xp.shape[1]  # number of particles
        n = mc  # number of voxels in sampling grid
        pwidth = int((math.ceil(epsP / h)))  # upper bound for support of basis functions

        # map particle positions to the domain [0, mc]
        xp = (xp - omega[0]) / h

        # get cell index of particles center of mass
        P = torch.ceil(xp)
        w = xp - (P - 1)

        B = (self.int1D_parallel(w, pwidth, epsP, h, hp, n_parallel)).reshape(n_parallel, -1, 1)
        J = torch.arange(0, np, device=self.device).repeat(n_parallel, 2 * pwidth + 1, 1).reshape(n_parallel, -1, 1)
        i0 = torch.repeat_interleave(n * torch.arange(0, n_parallel, device=self.device).view(-1, 1), 3 * n).reshape(n_parallel, 3 * n, 1)
        I = (P.unsqueeze(dim=1).repeat(1, 3, 1) + torch.arange(-pwidth - 1, pwidth, device=self.device).unsqueeze(dim=1).expand(P.shape[0], 3, P.shape[1])).reshape(n_parallel, -1, 1)
        valid = torch.logical_and(I >= 0, I < mc)
        I = I[valid].long()
        J = J[valid].long()
        B = B[valid]
        I = I + i0[valid]

        T = torch.zeros(n_parallel * n, np, dtype=B.dtype, device=self.device)
        T[I, J] = B

        return T.reshape(n_parallel, n, np)

    def int1D_parallel(self, w, pwidth, eps, h, hp, n_parallel):
        """
        One-dimensional interpolation for distortion correction.

        Parameters
        ----------
        w : torch.Tensor
            Input data.
        pwidth : int
            Upper bound for the support of basis functions.
        eps : float
            Particle width.
        h : float
            Cell-size in the distortion dimension.
        hp : float
            Cell-size in the distortion dimension.
        n_parallel : int
            Size of the non-distortion dimensions.

        Returns
        ----------
        Bij : torch.Tensor
            Interpolated data.
        """
        Bij = torch.zeros(n_parallel, 2 * pwidth + 1, w.shape[1], dtype=w.dtype, device=self.device)
        Bleft = self.B_parallel(-pwidth - w, eps, h, n_parallel)
        for p in range(-pwidth, pwidth + 1):
            Bright = self.B_parallel(1 + p - w, eps, h, n_parallel)
            Bij[:, p + pwidth, :] = hp * (Bright - Bleft).squeeze()
            Bleft = Bright
        return Bij

    def B_parallel(self, x, eps, h, n_parallel):
        """
        Indexing and combination for one-dimensional interpolation.

        Parameters
        ----------
        x : torch.Tensor
            input data
        eps : float
            particle width
        h : float
            cell-size in distortion dimension
        n_parallel : int
            size of non-distortion dimensions

        Returns
        ----------
        Bij : torch.Tensor
            interpolated data
        """
        Bij = torch.zeros(n_parallel, x.shape[1], dtype=x.dtype, device=self.device)
        ind1 = (-eps / h <= x) & (x <= 0)
        ind2 = (0 < x) & (x <= eps / h)
        ind3 = (eps / h < x)
        Bij[ind1] = x[ind1] + 1 / (2 * eps / h) * x[ind1] ** 2 + eps / (h * 2)
        Bij[ind2] = x[ind2] - 1 / (2 * eps / h) * x[ind2] ** 2 + eps / (h * 2)
        Bij[ind3] = eps / h
        return Bij / eps

    def int1DSingle(self, w, pwidth, eps, h, hp):
        """
        One-dimensional interpolation for distortion correction.

        Parameters
        ----------
        w : torch.Tensor
            Input data.
        pwidth : int
            Upper bound for the support of basis functions.
        eps : float
            Particle width.
        h : float
            Cell-size in the distortion dimension.
        hp : float
            Cell-size in the distortion dimension.

        Returns
        ----------
        Bij : torch.Tensor
            Interpolated data.
        """
        Bij = torch.zeros(2 * pwidth + 1, w.shape[0], dtype=w.dtype, device=self.device)
        Bleft = self.B_single(-pwidth - w, eps, h)
        for p in range(-pwidth, pwidth + 1):
            Bright = self.B_single(1 + p - w, eps, h)
            Bij[p + pwidth, :] = hp * (Bright - Bleft).squeeze()
            Bleft = Bright
        return Bij

    def B_single(self, x, eps, h):
        """
        Indexing and combination for one-dimensional interpolation.

        Parameters
        ----------
        x : torch.Tensor
            input data
        eps : float
            particle width
        h : float
            cell-size in distortion dimension
        n_parallel : int
            size of non-distortion dimensions

        Returns
        ----------
        Bij : torch.Tensor
            interpolated data
        """
        Bij = torch.zeros(x.shape[0], dtype=x.dtype, device=self.device)
        ind1 = (-eps / h <= x) & (x <= 0)
        ind2 = (0 < x) & (x <= eps / h)
        ind3 = (eps / h < x)
        Bij[ind1] = x[ind1] + 1 / (2 * eps / h) * x[ind1] ** 2 + eps / (h * 2)
        Bij[ind2] = x[ind2] - 1 / (2 * eps / h) * x[ind2] ** 2 + eps / (h * 2)
        Bij[ind3] = eps / h
        return Bij / eps
