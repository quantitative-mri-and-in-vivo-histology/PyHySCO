import math
from EPI_MRI.utils import *


class LeastSquaresCorrectionMultiPe:
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

        # Process each PE-RPE pair
        for i, pair in enumerate(self.dataObj.image_pairs):

            v = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dataObj.dtype)
            P = torch.eye(3, device=self.dataObj.device, dtype=self.dataObj.dtype)[self.dataObj.permute[i]]
            T = torch.tensor(self.dataObj.rel_mats[i][:3, :3], device=self.dataObj.device, dtype=self.dataObj.dtype)
            T_permuted = P @ T @ P.T

            v_rot = T_permuted @ v


            bc1 = self.A.mat_mul(yc).reshape(-1, 1)  # averaging matrix & translation vector
            bc1_full = bc1.reshape(-1,1)*v_rot.view(1, -1)

            # --- NEW: Get full coordinate grid (all dims) ---
            xc = get_cell_centered_grid(self.dataObj.omega, self.dataObj.m, device=self.device, dtype=self.dataObj.dtype, return_all=True)
            xc_rot = T_permuted @ xc.view(3,-1)

            xc_rot = xc_rot[1::, :]
            bc1_full = bc1_full[:, 1::]

            xp1 = xc_rot + bc1_full.T        
            xp2 = xc_rot - bc1_full.T

            xp1 = xp1.reshape(2, *self.dataObj.m)
            xp2 = xp2.reshape(2, *self.dataObj.m)

            rho0 = pair.pe_image
            rho1 = pair.rpe_image

            rho0 = rho0.reshape(self.dataObj.m[0], -1)
            rho1 = rho1.reshape(self.dataObj.m[0], -1)

            # Get push-forward matrices for this pair
            C1 = self.get_push_forward_matrix_2d(self.dataObj.omega, xp1.clone(), self.dataObj.h, self.dataObj.h)
            C2 = self.get_push_forward_matrix_2d(self.dataObj.omega, xp2.clone(), self.dataObj.h, self.dataObj.h)
            C = torch.cat((C1, C2), dim=1)
            
            # Store matrices and data
            C_list.append(C)
            rho_list.append(torch.hstack((rho0, rho1)))

        # Concatenate all matrices and data
        C_all = torch.cat(C_list, dim=1)
        rho_all = torch.cat(rho_list, dim=1)

        # Solve least squares problem
        rhocorr = torch.linalg.lstsq(C_all, rho_all.unsqueeze(2)).solution
        return rhocorr.reshape(list(self.dataObj.m))


    def get_push_forward_matrix_2d(self, omega, xp, h, hp, device=None, dtype=torch.float32):
        """
        Construct a dense push-forward matrix for bilinear interpolation in 2D slices.

        Parameters
        ----------
        xp : torch.Tensor, shape (2, D, H, W)
            Particle positions (x, y) for each slice.

        Returns
        -------
        T_all : torch.Tensor, shape (D * H * W, H * W)
            Dense matrix mapping particle weights to voxel grid.
        """
        if device is None:
            device = xp.device

        _, D, H, W = xp.shape
        N = H * W  # number of particles per slice
        T_slices = []

        for d in range(D):
            x = xp[0, d].reshape(-1)
            y = xp[1, d].reshape(-1)

            x = (x - omega[0]) / h[0]
            y = (y - omega[2]) / h[1]

            x0 = torch.floor(x).long()
            y0 = torch.floor(y).long()
            x1 = x0 + 1
            y1 = y0 + 1

            x0 = x0.clamp(0, W - 1)
            x1 = x1.clamp(0, W - 1)
            y0 = y0.clamp(0, H - 1)
            y1 = y1.clamp(0, H - 1)

            wx = x - x0.float()
            wy = y - y0.float()

            w00 = (1 - wx) * (1 - wy)
            w10 = wx * (1 - wy)
            w01 = (1 - wx) * wy
            w11 = wx * wy

            def flatten(y_idx, x_idx):
                return y_idx * W + x_idx

            i00 = flatten(y0, x0)
            i10 = flatten(y0, x1)
            i01 = flatten(y1, x0)
            i11 = flatten(y1, x1)

            particle_indices = torch.arange(N, device=device).repeat(4)
            voxel_indices = torch.cat([i00, i10, i01, i11], dim=0)
            weights = torch.cat([w00, w10, w01, w11], dim=0)

            T = torch.zeros(N, N, dtype=dtype, device=device)
            T[voxel_indices, particle_indices] = weights

            T_slices.append(T)

        T_all = torch.stack(T_slices, dim=0)  # shape (D, H * W, H * W)
        return T_all


    # def get_push_forward_matrix_2d(self, omega, xp, h, hp, device=None, dtype=torch.float32):
    #     """
    #     Construct a dense push-forward matrix for bilinear interpolation in 2D slices.

    #     Parameters
    #     ----------
    #     xp : torch.Tensor, shape (2, D, H, W)
    #         Particle positions (x, y) for each slice.
        
    #     Returns
    #     -------
    #     T : torch.Tensor, shape (M_total, N)
    #         Dense matrix mapping particle weights to affected voxel grid.
    #     voxel_indices : torch.Tensor
    #         Linear indices of the affected output voxels.
    #     """
    #     if device is None:
    #         device = xp.device

    #     _, D, H, W = xp.shape
    #     N = H * W  # total number of particles

    #     # Flatten all positions
    #     x = xp[0].reshape(-1)
    #     y = xp[1].reshape(-1)

    #     x = x.reshape(D, -1)
    #     y = y.reshape(D, -1)

    #     # Integer neighbors
    #     x0 = torch.floor(x).long()
    #     y0 = torch.floor(y).long()
    #     x1 = x0 + 1
    #     y1 = y0 + 1

    #     # Clip to bounds
    #     x0 = x0.clamp(0, W - 1)
    #     x1 = x1.clamp(0, W - 1)
    #     y0 = y0.clamp(0, H - 1)
    #     y1 = y1.clamp(0, H - 1)

    #     # Bilinear weights
    #     wx = x - x0.float()
    #     wy = y - y0.float()

    #     w00 = (1 - wx) * (1 - wy)
    #     w10 = wx * (1 - wy)
    #     w01 = (1 - wx) * wy
    #     w11 = wx * wy

    #     def flatten(y, x):
    #         return y * W + x

    #     # Compute destination indices (flattened across D, H, W)
    #     i00 = flatten(y0, x0)
    #     i10 = flatten(y0, x1)
    #     i01 = flatten(y1, x0)
    #     i11 = flatten(y1, x1)

    #     # Repeat particle indices for 4 weights
    #     particle_indices = torch.arange(N, device=device)  # shape [D*N]
    #     particle_indices = particle_indices.repeat(4)          # shape [4 * D*N]

    #     # Stack all weights and indices
    #     voxel_indices_all = torch.cat([i00, i10, i01, i11], dim=0)
    #     weights_all = torch.cat([w00, w10, w01, w11], dim=0)

    #     # Create reduced dense matrix
    #     T = torch.zeros(xp.shape[1]*xp.shape[2], N, dtype=dtype, device=device)
    #     T[voxel_indices_all, particle_indices] = weights_all

    #     return T


    # def get_push_forward_matrix_2d(self, omega, xp, h, hp, device=None, dtype=torch.float32):
    #     """
    #     Construct a dense push-forward matrix for bilinear interpolation in 2D slices.

    #     Parameters
    #     ----------
    #     xp : torch.Tensor, shape (2, D, H, W)
    #         Particle positions (x, y) for each slice.
        
    #     Returns
    #     -------
    #     T : torch.Tensor, shape (M_total, N)
    #         Dense matrix mapping particle weights to affected voxel grid.
    #     voxel_indices : torch.Tensor
    #         Linear indices of the affected output voxels.
    #     """
    #     if device is None:
    #         device = xp.device

    #     _, D, H, W = xp.shape
    #     N = D * H * W  # total number of particles

    #     # Build grid of (z, y, x) indices for flattening
    #     z_grid = torch.arange(D, device=device).repeat_interleave(H * W)
    #     y_grid = torch.arange(H, device=device).repeat(W).repeat(D)
    #     x_grid = torch.tile(torch.arange(W, device=device), (H * D,))

    #     # Flatten all positions
    #     x = xp[0].reshape(-1)
    #     y = xp[1].reshape(-1)

    #     # Integer neighbors
    #     x0 = torch.floor(x).long()
    #     y0 = torch.floor(y).long()
    #     x1 = x0 + 1
    #     y1 = y0 + 1

    #     # Clip to bounds
    #     x0 = x0.clamp(0, W - 1)
    #     x1 = x1.clamp(0, W - 1)
    #     y0 = y0.clamp(0, H - 1)
    #     y1 = y1.clamp(0, H - 1)

    #     # Bilinear weights
    #     wx = x - x0.float()
    #     wy = y - y0.float()

    #     w00 = (1 - wx) * (1 - wy)
    #     w10 = wx * (1 - wy)
    #     w01 = (1 - wx) * wy
    #     w11 = wx * wy

    #     def flatten(z, y, x):
    #         return z * (H * W) + y * W + x

    #     # Compute destination indices (flattened across D, H, W)
    #     z = z_grid
    #     i00 = flatten(z, y0, x0)
    #     i10 = flatten(z, y0, x1)
    #     i01 = flatten(z, y1, x0)
    #     i11 = flatten(z, y1, x1)

    #     # Repeat particle indices for 4 weights
    #     particle_indices = torch.arange(N, device=device)
    #     particle_indices = particle_indices.repeat(4)

    #     # Stack all weights and indices
    #     voxel_indices_all = torch.cat([i00, i10, i01, i11], dim=0)
    #     weights_all = torch.cat([w00, w10, w01, w11], dim=0)

    #     # Reduce to unique affected voxel indices
    #     unique_voxels, inverse_map = torch.unique(voxel_indices_all, return_inverse=True)
    #     M = unique_voxels.shape[0]

    #     # Create reduced dense matrix
    #     T = torch.zeros(M, N, dtype=dtype, device=device)
    #     T[inverse_map, particle_indices] = weights_all

    #     return T


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
