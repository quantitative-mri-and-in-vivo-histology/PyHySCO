import torch
from triton.language import dtype

from EPI_MRI.ImageModels import *
from EPI_MRI.InitializationMethods import *
from EPI_MRI.Preconditioners import *
import torchsparsegradutils as tsgu
#import jax
from torchsparsegradutils.jax import sparse_solve_j4t
#from torchsparsegradutils.cupy import sparse_solve_c4t
from EPI_MRI.ParticleInCell2D import *
from optimization.LinearSolvers import *


class EPIMRIDistortionCorrectionPush4dProper:
    """
    Implements distortion correction using a least squares approach to find a single corrected image
    from multiple phase encoding directions.

    Parameters
    ----------
    data_obj : `DataObject`
        object containing interpolation models for input images along with domain size and details
    alpha : float
        regularization parameter for the smoothness regularizer
    beta : float
        regularization parameter for the intensity regularizer
    averaging_operator : Class (subclass of :obj:`LinearOperators.LinearOperator`), optional
        class to use for the averaging operator (default is `LinearOperators.myAvg1D`)
    derivative_operator : Class (subclass of `LinearOperators.LinearOperator`), optional
        class to use for the derivative operator (default is `LinearOperators.myDiff1D`)
    regularizer : Class (subclass of `.LinearOperators.LinearOperator`), optional
        class to use for the regularizer (default is `LinearOperators.myLaplacian3D`)
    rho : float, optional
        augmentation parameter for proximal term (default is 0.0)
    initialization : Class (subclass of `InitializationMethods.InitializationMethod`), optional
        class to use for the initialization of the field map (default is `InitializationMethods.InitializeCF`)
    PC : Class (subclass of `Preconditioners.Preconditioner`), optional
        preconditioner (default is `Preconditioners.JacobiCG`)

    Attributes
    ----------
    dataObj : `DataObject`
        object containing interpolation models for input images along with domain size and details
    A : `LinearOperators.LinearOperator`
        averaging operator
    D : `LinearOperators.LinearOperator`
        partial derivative operator in the phase encoding dimension
    xc : torch.Tensor (size prod(m))
        cell-centered grid in the phase encoding dimension
    S : `Regularizers.QuadRegularizer`
        defines the smoothness regularizer
    Q : `Regularizers.TikRegularizer`
        defines the proximal term, if used
    PC : `Preconditioners.Preconditioner`
        preconditioner object, if used
    initialization : `InitializationMethods.InitializationMethod`
        initialization object
    alpha : float
        regularization parameter for the smoothness regularizer
    beta : float
        regularization parameter for the intensity regularizer
    rho : float
        parameter for the proximal term augmentation parameter, if used
    device : string
        device on which to compute operations
    dtype : torch.dtype
        data type for all data tensors
    Dc : float
        most recent data fit term value
    Sc : float
        most recent smoothness regularization term value
    Pc : float
        most recent intensity regularization term value
    Qc : float
        most recent proximal term value, if used
    corr1 : torch.Tensor (size m)
        most recent corrected image from dataObj.I1
    corr2 : torch.Tensor (size m)
        most recent corrected image from dataObj.I2

    """

    def __init__(self,
                 data_obj,
                 alpha,
                 beta,
                 recon_size=None,
                 averaging_operator=myAvg1D,
                 derivative_operator=myDiff1D,
                 regularizer=myLaplacian3D,
                 rho=0.0,
                 lambda_smooth=0.001,
                 initialization=InitializeCF,
                 PC=JacobiCG):
        self.dataObj = data_obj
        self.device = data_obj.device
        self.dtype = data_obj.dtype
        self.recon_image = None

        self.m_distorted = self.dataObj.m
        self.omega_distorted = self.dataObj.omega
        self.h_distorted = self.dataObj.h

        if recon_size is None:
            self.m_recon = self.dataObj.m
        else:
            self.m_recon = recon_size
            self.omega_recon = self.omega_distorted
            self.h_recon = self.h_distorted / self.m_distorted * self.m_recon

        self.A = averaging_operator(self.omega_recon[2:], self.m_recon[1:],
                                    self.dtype, self.device)
        self.D_slice = derivative_operator(self.omega_recon[-4:],
                                           self.m_recon[-2:], self.dtype,
                                           self.device)
        self.D_image = derivative_operator(self.omega_recon[-4:],
                                           self.m_recon[-2:], self.dtype,
                                           self.device)
        self.D = derivative_operator(self.omega_recon[-6:], self.m_recon[-3:],
                                     self.dtype, self.device)
        self.xc = get_cell_centered_grid(self.omega_recon, self.m_recon,
                                         device=self.device,
                                         dtype=self.dtype).reshape(
            tuple(self.m_recon))
        self.S = QuadRegularizer(
            regularizer(self.omega_recon[2:], self.m_recon[1:], self.dtype,
                        self.device))
        self.Q = TikRegularizer(self.omega_recon, self.m_recon)
        self.alpha = alpha
        self.beta = beta
        self.lambda_smooth = lambda_smooth
        self.rho = rho
        self.initialization = initialization()
        self.Dc = None
        self.Sc = None
        self.Pc = None
        self.Qc = None
        self.corr1 = None
        self.corr2 = None
        if PC is not None:
            self.PC = PC(self.dataObj)
        else:
            self.PC = None
        self.solver = PCG(max_iter=20, tol=1e-5, verbose=False)
        # --- NEW: Get full coordinate grid (all dims) ---
        self.base_grid = get_cell_centered_grid(self.omega_recon[2:],
                                                self.m_recon[1:],
                                                device=self.device,
                                                dtype=self.dataObj.dtype,
                                                return_all=True)

        images = []
        pe_dirs = []
        bvec = []
        bval = []
        rel_mats = []
        masks = []

        for i, pair in enumerate(self.dataObj.image_pairs):
            pe_image = pair[0]
            rpe_image = pair[1]
            pe_mask = pair[0].mask
            rpe_mask = pair[1].mask

            pe_bvec = torch.tensor(pe_image.bvec, dtype=self.dtype,
                                   device=self.device)
            rpe_bvec = torch.tensor(rpe_image.bvec, dtype=self.dtype,
                                    device=self.device)
            pe_bval = torch.tensor(pe_image.bval, dtype=self.dtype,
                                   device=self.device)
            rpe_bval = torch.tensor(rpe_image.bval, dtype=self.dtype,
                                    device=self.device)

            pe_dti_volumes = pe_image.data
            rpe_dti_volumes = rpe_image.data

            masks.append(pe_mask)
            masks.append(rpe_mask)
            images.append(pe_dti_volumes)
            images.append(rpe_dti_volumes)
            bvec.append(pe_bvec.T)
            bvec.append(rpe_bvec.T)
            bval.append(pe_bval)
            bval.append(rpe_bval)
            pe_dirs.append(torch.tensor(pe_image.phase_sign, dtype=self.dtype,
                                        device=self.device))
            pe_dirs.append(torch.tensor(rpe_image.phase_sign, dtype=self.dtype,
                                        device=self.device))
            rel_mats.append(self.dataObj.rel_mats[i])
            rel_mats.append(self.dataObj.rel_mats[i])

        self.images = torch.stack(images, dim=0)
        self.masks = torch.stack(masks, dim=0)
        self.bvec = torch.stack(bvec, dim=0)
        self.bval = torch.stack(bval, dim=0)
        self.rel_mats = torch.stack(rel_mats, dim=0)
        self.pe_dirs = torch.tensor(pe_dirs, dtype=self.dtype,
                                    device=self.device)
        self.base_grid = get_cell_centered_grid(self.omega_recon[2:],
                                                self.m_recon[1:],
                                                device=self.device,
                                                dtype=self.dataObj.dtype,
                                                return_all=True)
        self.base_grid = self.base_grid.reshape(3, -1)

        self.n_particles = torch.prod(self.m_recon[2:]).long()
        self.n_cells = torch.prod(self.m_distorted[2:]).long()
        self.v_pe = torch.tensor([0.0, 0.0, 1.0], device=self.device,
                                 dtype=self.dataObj.dtype)
        self.image_center = 0.5 * (
                torch.tensor(self.omega_recon[3::2]) + torch.tensor(
            self.omega_recon[2::2]))  # (x_c, y_c, z_c)

        self.Lx, self.Ly = get_2d_gradient_matrix(
            self.m_recon[-2],
            self.m_recon[-1],
            self.device,
            self.dtype)
        self.LxT_Lx = self.Lx.T @ self.Lx
        self.LyT_Ly = self.Ly.T @ self.Ly

    # noinspection PyTupleAssignmentBalance
    def eval(self, yc, yref=None, do_derivative=False, calc_hessian=False):
        """
        Evaluates the objective function given the field map yc.

        J(yc) = D(I1(yc),I2(yc)) + alpha S(yc) + beta P(yc) + rho Q(yc, yref)

        If do_derivative is True, returns the gradient as well.

        If calc_hessian is True, returns the Hessian and PC as well.

        Parameters
        ----------
        yc : torch.Tensor (size m_plus(m))
            a field inhomogeneity map
        yref : torch.Tensor (size m_plus(m)), optional
            reference image used in proximal term, default is None
        do_derivative : boolean, optional
            flag to compute and return the gradient, default is False
        calc_hessian : boolean, optional
            flag to construct and return Hessian mat-vec, default is False

        Returns
        -------
        J(yc) : torch.Tensor (size 1)
            objective function value
        dJ(yc) : torch.Tensor (size m_plus(m))
            gradient of the objective function, only returned when do_derivative=True
        H : Callable
            callable matrix-vector product with (approximate) Hessian, only returned when calc_hessian=True
        PC : Callable
            callable solver to apply preconditioner, only returned when calc_hessian=True
        """
        # Initialize lists to store matrices and data

        self.recon_image, residuals, jlsq, dlsq = self.solve_lsq(yc)

        save_data(self.recon_image.permute(2, 3, 1, 0),
                  f"/home/laurin/workspace/PyHySCO/data/results/debug/recon.nii.gz")

        Dc = (0.5 * (residuals ** 2 * jlsq[:, None])).mean(dim=[0,1]).sum()
        grad = (dlsq * jlsq[:, None]).mean(dim=(0, 1))

        pad_w = yc.shape[-1] - grad.shape[-1]  # Should be 1
        # Pad at the beginning instead of the end
        dD = torch.nn.functional.pad(grad, (1, 0),
                                     mode='replicate')  # Now (66,67)

        dD_pre = self.D.transp_mat_mul(grad)

        # compute distance measure
        hd_source = torch.prod(self.h_recon)
        hd_target = torch.prod(self.h_distorted)

        # smoothness regularizer
        Sc, dS, d2S = self.S.eval(yc, do_derivative=do_derivative)

        # intensity regularizer
        dbc = self.D.mat_mul(yc)
        Jac = 1 + dbc  # determinant of the transform xc+bc
        G, dG, d2G = self.phi_EPI(Jac - 1, do_derivative=do_derivative,
                                  calc_hessian=calc_hessian)
        Pc = torch.sum(G)
        dP = None
        if do_derivative:
            dP = self.D.transp_mat_mul(dG)

        # compute proximal term
        if self.rho > 0:
            Qc, dQ, d2Q = self.Q.eval(yc, 1.0, yref,
                                      do_derivative=do_derivative)
        else:
            Qc = 0.0
            dQ = 0.0
            d2Q = None

        # save terms of objective function and corrected images
        self.Dc = Dc
        self.Sc = Sc
        self.Pc = Pc
        self.Qc = Qc

        geom = (dlsq * jlsq[:, None]).mean(dim=(0, 1))   # sensitivity wrt ∂b/∂x
        intensity = (residuals * jlsq[:, None]).mean(dim=(0, 1))

        Jc = hd_target * Dc + hd_source * self.alpha * Sc + hd_source * self.beta * Pc + self.rho * Qc
        if not do_derivative:
            return Jc
        dJ = hd_target * dD + hd_source * self.alpha * dS + hd_source * self.beta * dP + self.rho * dQ

        # save_data(dD.permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/dD.nii.gz")
        # save_data(dD_pre.permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/dDDiff.nii.gz")
        # save_data(dS.permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/dS.nii.gz")
        # save_data(dP.permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/dP.nii.gz")
        # save_data(grad.permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/grad.nii.gz")
        # save_data(geom.permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/geom.nii.gz")
        # save_data(intensity.permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/intensity.nii.gz")


        # if not calc_hessian:
        #     return Jc, dJ
        # else:
        #     def H(x):
        #         """ Matrix-vector product between Hessian and a tensor x of size m_plus(m). """
        #         Dx = self.D.mat_mul(x)
        #         dr = geom * Dx + intensity * self.A.mat_mul(x)
        #         dr_d2psi = dr * hd_source
        #         if self.beta == 0:  # d2P is zeros
        #             d2D = self.D.transp_mat_mul(
        #                 dr_d2psi * geom) + self.A.transp_mat_mul(
        #                 dr_d2psi * intensity)
        #             return d2D + hd_source * self.alpha * d2S.mat_mul(
        #                 x) + hd_source * self.rho * x
        #         else:
        #             d2D = self.D.transp_mat_mul(
        #                 dr_d2psi * geom + hd_source * self.beta * d2G * Dx) + self.A.transp_mat_mul(
        #                 dr_d2psi * intensity)
        #             return d2D + hd_source * self.alpha * d2S.mat_mul(
        #                 x) + hd_source * self.rho * x
        #
        #     if self.PC is not None:
        #         diagD, diagP, diagS = self.PC.getM(geom, intensity, hd_source, d2G,
        #                                            self.D,
        #                                            self.A, self.S.H, self.alpha,
        #                                            self.beta)
        #         self.PC.M += hd_source * self.rho
        #         M = lambda x: self.PC.eval(x)
        #     else:
        #         M = None
        #
        #     return Jc, dJ, H, M

        # M = None
        #
        # Even more simplified version focusing only on geometric term:
        def H_geometric_only(x):
            """
            Alternative: Focus only on geometric deformation effects.
            This might be more appropriate for EPI correction.
            """
            # Dx = self.D.mat_mul(x)
            # dr_geom = geom * Dx * hd_target
            #
            # # Only geometric coupling
            # d2D_geom = self.D.transp_mat_mul(dr_geom * geom)
            #
            # # Add regularization
            # return d2D_geom + hd_source * self.alpha * d2S.mat_mul(
            #     x) + self.rho * x

            Dx = self.D.mat_mul(x)
            dr = geom * Dx
            dr_d2psi = dr * hd_source
            d2D = self.D.transp_mat_mul(
                dr_d2psi * geom)
            return hd_target * d2D + hd_source * self.alpha * d2S.mat_mul(x)

        # # assert isinstance(D, Conv1D)
        # #
        # For geometric-only case, we only need D^T D term
        D2 = self.D.op_mul(self.D)

        # Geometric data fidelity term: only geom^2 * D^T D
        diagD = D2.transp_mat_mul(geom ** 2)

        # Smoothness term (unchanged)
        diagS = self.S.H.diag()

        # Build preconditioner (no intensity or beta terms)
        M = hd_source * diagD + hd_source * self.alpha * diagS

        def apply_M(x):
            return x / M

        return Jc, dJ, H_geometric_only, apply_M




    def solve_lsq(self, yc):
        bc = self.A.mat_mul(yc).reshape(-1,1)  # averaging matrix & translation vector
        bc_3d = (bc * self.v_pe.view(1, -1)).T  # shift vector in original space
        self.bc = bc

        recon_vols = []
        jac = torch.zeros((self.images.shape[0], *self.m_recon[1:]), dtype=self.dtype, device=self.device)
        jacobians = torch.zeros((*self.images.shape[0:2], *self.m_recon[1:]), dtype=self.dtype, device=self.device)
        residuals = torch.zeros((*self.images.shape[0:2], *self.m_recon[1:]), dtype=self.dtype, device=self.device)

        for slice_index in range(self.images.shape[2]):

            stencil_weights = []
            stencil_indices = []
            d_images = []
            diag = torch.zeros(self.n_particles, device=self.device)

            for pe_dir_index in range(self.images.shape[0]):
                xp = self.base_grid + self.pe_dirs[
                    pe_dir_index] * bc_3d
                rot_mat_permuted = torch.linalg.inv(self.rel_mats[pe_dir_index, :3, :3])
                # rot_mat_permuted = self.rel_mats[pe_dir_index, :3, :3]
                xp = rot_mat_permuted @ (xp - self.image_center.unsqueeze(
                    1)) + self.image_center.view(3, 1)
                xp = xp[1:, :].reshape(2, *self.m_recon[1:])

                indices, weights, d_image, jac_slice = build_pic_stencils_2d(
                    self.omega_recon[-4:],
                    self.m_recon[-2:],
                    self.m_distorted[-2:],
                    xp[:, slice_index, :, :],
                    do_derivative=True,
                    return_jacobian=True)

                jac[pe_dir_index,slice_index] += jac_slice.reshape(*self.m_recon[2:])

                stencil_indices.append(indices)
                stencil_weights.append(weights)
                d_images.append(d_image)

                diag += build_diag_preconditioner(indices, weights, self.n_particles)

            stencil_indices = torch.stack(stencil_indices, dim=0)
            stencil_weights = torch.stack(stencil_weights, dim=0)

            recon_vol_slices = []
            for vol_index in range(self.images.shape[1]):

                recon_vol_slice, res, dC = self.solve_lsq_slice(vol_index, slice_index, stencil_indices, stencil_weights, d_images, diag)
                recon_vol_slices.append(recon_vol_slice)
                residuals[:,vol_index,slice_index] = res.reshape(-1, *self.m_recon[2:])
                jacobians[:,vol_index,slice_index] = dC.reshape(-1, *self.m_recon[2:])

            recon_vols.append(torch.stack(recon_vol_slices, dim=0))

        recon_image = torch.stack(recon_vols, dim=0)
        recon_image = recon_image.reshape(*recon_image.shape[0:2], *self.m_recon[2:])
        recon_image = recon_image.permute(1, 0, 2, 3)

        return recon_image, residuals, jac, jacobians


    def solve_lsq_slice(self, vol_index, slice_index, stencil_indices, stencil_weights, derivative_ops, diag):

        pe_slices = self.images[:, vol_index, slice_index]
        pe_slice_masks = self.masks[:, slice_index]
        pe_slices = pe_slices.view(pe_slices.size(0), -1)
        pe_slice_masks = pe_slice_masks.view(pe_slice_masks.size(0), -1)
        pe_slice_masks_flat = pe_slice_masks.reshape(-1)

        stencil_indices_masked = stencil_indices.view(-1, stencil_indices.shape[-1])
        stencil_weights_masked = stencil_weights.view(-1, stencil_indices.shape[-1])
        pe_slices_masked = pe_slices

        # stencil_indices_masked = stencil_indices.view(-1, stencil_indices.shape[-1])[pe_slice_masks_flat]
        # stencil_weights_masked = stencil_weights.view(-1, stencil_indices.shape[-1])[pe_slice_masks_flat]
        # pe_slices_masked = pe_slices[pe_slice_masks]


        def A_fn(x):
            """
            A(x) = Σ_k T_kᵀ T_k x    with x on N particles.
            Returns (N,1)   – no mixing between observations.
            """
            n_obs, N, K = stencil_indices.shape
            device = x.device
            Ax = torch.zeros_like(x)  # (N,1)
            Axtest = torch.zeros(self.n_cells, 1, device=device, dtype = torch.float32)  # (N,1)

            for k in range(n_obs):
                idx_k = stencil_indices[k]  # (N,K)
                w_k = stencil_weights[k]  # (N,K)

                # ---------- forward scatter: y_k = T_k x -----------------
                vals = (x * w_k).reshape(-1, 1)  # (N*K,1)  ← no .expand()
                flat_idx = idx_k.reshape(-1)  # (N*K,)

                y_grid = torch.zeros((self.n_cells, 1), device=device)
                y_grid.scatter_add_(0, flat_idx.unsqueeze(1), vals)  # (C,1)


                # ---------- adjoint gather: z_k = T_kᵀ y_k --------------
                z_k = (y_grid[idx_k].squeeze(-1) * w_k).sum(dim=1,
                                                            keepdim=True)  # (N,1)
                Ax += z_k
                Axtest += y_grid

            # Regularization: lambda * (Lxᵗ Lx + Lyᵗ Ly) x
            if self.lambda_smooth > 0:
                # tikhonov_weight = 0.01
                # result += tikhonov_weight * x

                LTLx = self.LxT_Lx @ x + self.LyT_Ly @ x
                Ax += self.lambda_smooth * LTLx

            return Ax / n_obs


        pe_masked = pe_slices * pe_slice_masks

        def build_rhs(y_grid):
            """
            Right-hand side for PCG when x lives on particles.

            Parameters
            ----------
            y_grid : torch.Tensor
                Observed data on the grid, shape (n_obs, n_cells, 1)
                ─ or ─ (n_obs, n_cells) if you keep the last dim implicit.

            Returns
            -------
            b : torch.Tensor, shape (N, 1)
                b = Σ_k T_kᵀ y_k   (adjoint-interpolated observations)
            """
            n_obs, N, K = stencil_indices.shape  # n_obs = 8
            device = y_grid.device
            b = torch.zeros((N, 1), device=device)

            # ---------------------------------------------------------------
            # loop over observations; K is tiny so this is cheap
            # ---------------------------------------------------------------
            for k in range(n_obs):
                idx_k = stencil_indices[k]  # (N, K)
                w_k = stencil_weights[k]  # (N, K)

                # gather grid values that stencil k references
                y_k = y_grid[k]  # (n_cells, 1) or (n_cells,)
                y_gathered = y_k[idx_k]  # (N, K, 1) or (N, K)

                # adjoint interpolation: sum_j  w_ij · y(idx_ij)
                if y_gathered.dim() == 3:  # (N,K,1) → (N,K)
                    y_gathered = y_gathered.squeeze(-1)

                b_k = (y_gathered * w_k).sum(dim=1, keepdim=True)  # (N, 1)
                b += b_k  # accumulate

            return b / n_obs

        rhs = build_rhs(pe_masked)
        # rhs /= pe_masked.shape[0]
        # rhs = rhs.view(-1,1)

        # save_data(rhs.reshape(-1, *self.m_recon[2:]),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/rhs.nii.gz")
        #
        # save_data(pe_masked.reshape(-1, *self.m_distorted[2:]),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/pe_masked.nii.gz")

        if self.recon_image is None:
            x0 = torch.zeros(self.n_particles, 1, dtype=self.dtype, device=self.device)
        else:
            x0 = self.recon_image[vol_index,slice_index].view(-1, 1)
            # x0 = torch.zeros_like(rhs)
            # x0 = torch.zeros(self.n_particles, 1, dtype=self.dtype, device=self.device)

        def make_diag_preconditioner(diag):
            diag = diag.view(-1, 1)  # ensure it's column vector

            def M(x):
                x = x.view(-1, 1)  # ensure x is also column vector
                return x / (diag + 0.01 * diag.mean())  # safe: [N, 1] / [N, 1]

            return M

        M = make_diag_preconditioner(diag)
        rhocorr, _, _, _, _ = self.solver.eval(A=A_fn, b=rhs, x=x0)

        # save_data(rhocorr.reshape(-1, *self.m_recon[2:]),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/rhocorr.nii.gz")


        gathered = rhocorr[stencil_indices].squeeze(-1)  # [N_particles, K, B]
        weighted = gathered * stencil_weights # [N_particles, K, B]
        interpolated = weighted # [N_particles, B]
        # residuals = interpolated-pe_slices

        flat_idx = stencil_indices.reshape(-1)  # (N*K,)
        flat_contr = weighted.reshape(-1, 1)  # (N*K, 1)

        valid = (flat_idx >= 0) & (flat_idx < self.n_cells)
        flat_idx = flat_idx[valid]
        flat_contr = flat_contr[valid]

        res_k = torch.zeros((self.n_cells, 1), device=self.device)
        res_k.scatter_add_(0, flat_idx.unsqueeze(1),
                           flat_contr)  # (n_cells, 1)

        residuals = rhocorr-rhs


        dT_imgs = []
        for pe_index in range(pe_slices.shape[0]):
            d = derivative_ops[0](rhocorr)
            dx_weighted = residuals[pe_index]*d[:,0]
            dy_weighted = residuals[pe_index]*d[:,1]
            v_rot = self.v_pe
            v_inplane = v_rot[1:]
            v_inplane = v_inplane / torch.norm(v_inplane)
            dy_contrib = dx_weighted * v_inplane[0] + dy_weighted * v_inplane[1]
            dT_imgs.append(dy_contrib.reshape(*self.m_recon[2:]))
        dT_imgs = torch.stack(dT_imgs, dim=0)

        return rhocorr, residuals, dT_imgs


    def phi_EPI(self, x, do_derivative=False, calc_hessian=False):
        """
        Barrier function for the intensity regularization term, applied element-wise.

        phi(x) = -x^4 / ((x^2 - 1))

        phi satisfies these important conditions:

            * phi(x) > 0, for all x

            * phi(|x| -> 1) -> infinity

            * phi(0) = 0

            * phi is convex

            * phi(x) = phi(-x)

        Parameters
        ----------
        x : torch.Tensor (size m)
            partial derivative of the field map
        do_derivative : boolean, optional
            flag to compute the first derivative (default is False)
        calc_hessian : boolean, optional
            flag to compute the second derivative (default is False)

        Returns
        ----------
        G : torch.Tensor (size m)
            function value
        dG : torch.Tensor (size m)
            first derivative of the function, None if do_derivative=False
        d2G : torch.Tensor (size m)
            second derivative of the function, None if calc_hessian=False
        """
        dG, d2G = None, None
        # penalize values outside of (-1,1)
        x[torch.abs(x) >= 1] = float('inf')
        x2 = x * x
        G = torch.nan_to_num(-(x2 * x2) / (x2 - 1))
        if do_derivative:
            dG = torch.nan_to_num(-2 * (x * x2) * (x2 - 2) / (x2 - 1) ** 2)
        if calc_hessian:
            d2G = torch.nan_to_num(
                -2 * x2 * (x2 * x2 - 3 * x2 + 6) / (x2 - 1) ** 3)
        return G, dG, d2G

    def initialize(self, *args, **kwargs):
        """
        Calls the initialization scheme.

        Parameters
        ----------
        args, kwargs : Any
            arguments and keyword arguments as needed for the initialization scheme

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            initial guess for the field map
        """
        return self.initialization.eval(self.dataObj, *args, **kwargs)


def get_2d_gradient_matrix(H, W, device=None, dtype=torch.float32):
    """
    Returns Lx and Ly: sparse gradient operators in x and y directions for 2D image of size H x W
    I_flat is assumed to be flattened in row-major (C-style) order: I[y, x] → I[y * W + x]
    """

    # Convert to Python ints if they're torch.Tensors
    if isinstance(H, torch.Tensor):
        H = int(H.item())
    if isinstance(W, torch.Tensor):
        W = int(W.item())

    N = H * W
    idx = torch.arange(N, device=device)

    # Gradient in x-direction (horizontal)
    mask_x = (idx + 1) % W != 0  # exclude last column
    rows_x = torch.arange(mask_x.sum().item(), device=device)
    cols_x = idx[mask_x]
    cols_x_right = cols_x + 1
    data_x = torch.stack([-torch.ones_like(rows_x), torch.ones_like(rows_x)])
    indices_x = torch.stack(
        [rows_x.repeat(2), torch.cat([cols_x, cols_x_right])])
    Lx = torch.sparse_coo_tensor(indices_x, data_x.flatten(),
                                 size=(mask_x.sum().item(), N), device=device,
                                 dtype=dtype)

    # Gradient in y-direction (vertical)
    mask_y = idx < (H - 1) * W  # exclude last row
    rows_y = torch.arange(mask_y.sum(), device=device)
    cols_y = idx[mask_y]
    cols_y_down = cols_y + W
    data_y = torch.stack([-torch.ones_like(rows_y), torch.ones_like(rows_y)])
    indices_y = torch.stack(
        [rows_y.repeat(2), torch.cat([cols_y, cols_y_down])])
    Ly = torch.sparse_coo_tensor(indices_y, data_y.flatten(),
                                 size=(mask_y.sum().item(), N), device=device,
                                 dtype=dtype)

    return Lx.coalesce(), Ly.coalesce()


def build_pic_stencils_2d(omega, m_source, m_target, xp, do_derivative=False, return_jacobian=False):
    """
    Build interpolation stencils for matrix-free PIC push-forward operator.

    Parameters
    ----------
    omega : list or torch.Tensor of shape (4,)
        Domain bounds [x0, x1, y0, y1].
    m_target : tuple (H, W)
        Target grid size.
    xp : torch.Tensor of shape (2, N)
        Particle positions in physical space.
    pwidth : int
        Half-width of interpolation stencil.

    Returns
    -------
    indices : torch.LongTensor [N, (2p+1)^2]
        Flattened voxel indices per particle.
    weights : torch.FloatTensor [N, (2p+1)^2]
        Corresponding interpolation weights.
    dT : Callable (rho: [num_voxels]) → torch.sparse.Tensor [N, num_voxels, 2], optional
        Only returned if `do_derivative=True`.
        A callable that, for a given input image `rho`, returns the derivative of
        the interpolation operator with respect to particle positions (∂x, ∂y).
    Jac : torch.FloatTensor [N], optional
        Only returned if `return_jacobian=True`.
        Sum of interpolation weights per particle, representing the Jacobian
        determinant (mass-preserving factor) under the current deformation.
    """
    device = xp.device

    # source grid settings
    n_cells = torch.prod(m_source)
    cell_size = (omega[1::2] - omega[0::2]) / m_source

    # target grid settings
    n_particles = torch.prod(m_target)
    particle_size = (omega[1::2] - omega[0::2]) / m_target

    x_vox = (xp[0] - omega[0]) / particle_size[0]
    y_vox = (xp[1] - omega[2]) / particle_size[1]

    x_vox = x_vox
    y_vox = y_vox

    x_vox = x_vox.permute(1, 0)
    y_vox = y_vox.permute(1, 0)

    Px = torch.floor(x_vox).long()
    wx = x_vox - (Px.float())
    Py = torch.floor(y_vox).long()
    wy = y_vox - (Py.float())

    Px = Px.reshape(-1)
    Py = Py.reshape(-1)
    wx = wx.reshape(-1)
    wy = wy.reshape(-1)

    epsP = torch.tensor(1*particle_size, device=device, dtype=xp.dtype)
    # epsP = 4*cell_size
    pwidth = torch.ceil(epsP / cell_size).to(torch.int32)
    # pwidth = [3,3]

    # # # # # Evaluate 1D basis
    Bx, Dx = int1DSingle(wx, pwidth[0], epsP[0], cell_size[0],
                         particle_size[0], do_derivative=True)  # [2*p+1, N]
    By, Dy = int1DSingle(wy, pwidth[1], epsP[1], cell_size[1],
                         particle_size[1], do_derivative=True)

    # Bx = Bx / Bx.sum(0)
    # By = By / By.sum(0)

    nbx = Bx.shape[0]
    nby = By.shape[0]
    nVoxel = nbx * nby

    I = torch.empty(nVoxel * n_cells, dtype=torch.long, device=device)
    J = torch.empty(nVoxel * n_cells, dtype=torch.long, device=device)
    B = torch.empty(nVoxel * n_cells, dtype=xp.dtype, device=device)

    if do_derivative:
        dBx = torch.empty_like(B)
        dBy = torch.empty_like(B)

    pp = 0
    for i, px in enumerate(range(-pwidth[0], pwidth[0] + 1)):
        for j, py in enumerate(range(-pwidth[1], pwidth[1] + 1)):
            idx = slice(pp * n_cells, (pp + 1) * n_cells)
            pp += 1

            x_idx = Px + px
            y_idx = Py + py
            # Iij = x_idx * m_target[1] + y_idx
            Iij = y_idx * m_target[0] + x_idx
            Bij = Bx[i, :] * By[j, :]  # Elementwise per-particle weight

            I[idx] = Iij
            J[idx] = torch.arange(n_cells, device=Px.device)
            B[idx] = Bij

            if do_derivative:
                dBx[idx] = Dx[i, :] * By[j, :]
                dBy[idx] = Bx[i, :] * Dy[j, :]

    invalid_idx = (I < 0) | (I >= n_particles)
    valid_idx = ~invalid_idx
    I[invalid_idx] = 0
    J[invalid_idx] = 0
    B[invalid_idx] = 0

    stencil_size = nbx * nby

    indices = I.view(stencil_size, n_cells)
    weights = B.view(stencil_size, n_cells)
    indices = indices.permute(1, 0)
    weights = weights.permute(1, 0)
    jvals = J.view(stencil_size, n_cells)
    jvals = jvals.permute(1, 0)
    valid_idx_mask = valid_idx.view(stencil_size, n_cells)
    valid_idx_mask = valid_idx_mask.permute(1, 0)

    results = [indices, weights]

    if do_derivative:
        dBx[invalid_idx] = 0
        dBy[invalid_idx] = 0

        dBx = dBx.reshape(stencil_size, n_cells)
        dBy = dBy.reshape(stencil_size, n_cells)
        dBx = dBx.permute(1, 0)
        dBy = dBy.permute(1, 0)
        # dBx = dBx.view(-1)
        # dBy = dBy.view(-1)

        # def apply_dT_T(image):
        #     # Inputs: residual [n_cells, B] or [n_cells]
        #     if image.ndim == 1:
        #         image = image.unsqueeze(1)  # [n_cells, 1]
        #
        #     vals_x = (image[jvals].squeeze(-1) * dBx)
        #     vals_y = (image[jvals].squeeze(-1) * dBy)
        #
        #     ids = torch.arange(dBx.numel(), device=image.device)
        #     ids = ids[valid_idx]
        #
        #     vals_x = vals_x[valid_idx_mask]
        #     vals_y = vals_y[valid_idx_mask]
        #
        #     # vals_x = (residual[indices] * dBx).sum(dim=1)
        #     # vals_y = (residual[indices] * dBy).sum(dim=1)
        #
        #     g = torch.zeros((n_cells, 2), device=image.device)
        #     g[:, 0].scatter_add_(0, jvals[valid_idx_mask], vals_x / cell_size[0])
        #     g[:, 1].scatter_add_(0, jvals[valid_idx_mask], vals_y / cell_size[1])

            # return g.squeeze(1) # Sum over x/y for Gauss-Newton


        def apply_dT_T(residual):
            # Inputs: residual [n_cells, B] or [n_cells]
            if residual.ndim == 1:
                residual = residual.unsqueeze(1)  # [n_cells, 1]

            vals_x = residual[indices].squeeze(-1) * dBx # [K*N, B]
            vals_y = residual[indices].squeeze(-1) * dBy

            vals_x = (residual[indices].squeeze(-1) * dBx).sum(dim=1)
            vals_y = (residual[indices].squeeze(-1) * dBy).sum(dim=1)

            # vals_x = (residual[indices] * dBx).sum(dim=1)
            # vals_y = (residual[indices] * dBy).sum(dim=1)

            g = torch.zeros((n_cells, 2), device=residual.device)
            g[:, 0].scatter_add_(0,
                                 torch.arange(n_cells, device=residual.device),
                                 vals_x / cell_size[0])
            g[:, 1].scatter_add_(0,
                                 torch.arange(n_cells, device=residual.device),
                                 vals_y / cell_size[1])
            return g.squeeze(1) # Sum over x/y for Gauss-Newton

        # def apply_dT_T(image_particles):
        #     """
        #     Create sparse matrices representing ∂T/∂x * image and ∂T/∂y * image
        #
        #     Returns: Two sparse matrices of size [n_cells, n_particles] each
        #     """
        #     if image_particles.ndim == 1:
        #         image_particles = image_particles.unsqueeze(1)
        #
        #     # Get image values for each stencil contribution
        #     image_vals = image_particles[indices].squeeze(
        #         -1)  # [n_stencil_points]
        #
        #     # Apply derivative weights
        #     vals_x = (image_vals * dBx / cell_size[0])[valid_idx_mask]
        #     vals_y = (image_vals * dBy / cell_size[1])[valid_idx_mask]
        #
        #     # Get valid indices
        #     jvals_valid = jvals[valid_idx_mask]  # Cell indices (rows)
        #     ivals_valid = indices[valid_idx_mask]  # Particle indices (columns)
        #
        #     # Create sparse matrices [n_cells, n_particles]
        #     dT_x = torch.sparse_coo_tensor(
        #         torch.stack([jvals_valid, ivals_valid]),
        #         vals_x,
        #         size=(n_cells, n_particles),
        #         device=image_particles.device
        #     )
        #
        #     dT_y = torch.sparse_coo_tensor(
        #         torch.stack([jvals_valid, ivals_valid]),
        #         vals_y,
        #         size=(n_cells, n_particles),
        #         device=image_particles.device
        #     )
        #
        #     return dT_x, dT_y

        # def apply_dT_T(residuals):
        #     """
        #     Compute ∂T/∂x * image and ∂T/∂y * image
        #
        #     Args:
        #         image_particles: [n_particles] - image values on particles
        #
        #     Returns:
        #         [n_cells, 2] - but this should really be [n_cells, n_particles, 2] to match MATLAB
        #     """
        #     # Apply derivative weights
        #     vals_x = residuals.unsqueeze(-1) * dBx / cell_size[0]  # [n_stencil_points]
        #     vals_y = residuals.unsqueeze(-1) * dBy / cell_size[1]  # [n_stencil_points]
        #
        #     vals_x = vals_x.sum(-1)
        #     vals_y = vals_y.sum(-1)
        #
        #     return torch.stack([vals_x, vals_y], dim=0)  # [n_cells, 2]

        results.append(apply_dT_T)

    Jac = torch.zeros(n_cells, device=device)
    Jac = Jac.scatter_add(0, J[~invalid_idx], B[~invalid_idx])
    # B = B / (Jac[J] + 1e-6)
    if return_jacobian:
        results.append(Jac)

    return tuple(results) if len(results) > 1 else results[0]


def apply_pic_matvec(stencils, x):
    indices, weights = stencils  # [N, K]
    mask = (indices >= 0) & (indices < x.shape[0])  # validity mask

    safe_indices = indices.clone()
    safe_indices[~mask] = 0  # avoids out-of-bounds, but zeroed

    safe_weights = weights.clone()
    safe_weights[~mask] = 0

    x_gathered = x[safe_indices]  # [N, K, C]
    #x_gathered[~mask.unsqueeze(-1)] = 0  # zero out invalid weights

    return (x_gathered * safe_weights.unsqueeze(-1)).sum(dim=1)


def apply_pic_rmatvec(stencils, r, num_voxels):
    indices, weights = stencils
    out = torch.zeros((num_voxels, 1), device=r.device)

    for k in range(weights.shape[1]):
        idx_k = indices[:, k]     # [N]
        w_k = weights[:, k]       # [N]
        contrib = r * w_k.view(-1, 1)  # [N, 1]
        out.scatter_add_(0, idx_k.view(-1, 1), contrib)

    return out  # [num_voxels, 1]


def apply_pic_normal_matvec(indices, weights, x, num_voxels):
    """
    Vectorized version of Aᵀ A x (scalar case).

    Parameters
    ----------
    stencils : tuple of (indices [N, K], weights [N, K])
    x : torch.Tensor of shape [num_voxels]
    num_voxels : int

    Returns
    -------
    torch.Tensor of shape [num_voxels]
    """
    x_gathered = x[indices]  # [N, K, 1]
    Ax = (x_gathered * weights.unsqueeze(-1)).sum(dim=1)  # [N, 1]

    # Now back-project: multiply each Ax by weights and scatter
    out = torch.zeros((num_voxels, 1), device=x.device)

    for k in range(weights.shape[1]):
        idx_k = indices[:, k]     # [N]
        w_k = weights[:, k]       # [N]
        contrib = Ax * w_k.view(-1, 1)  # [N, 1]
        out.scatter_add_(0, idx_k.view(-1, 1), contrib)

    return out  # [num_voxels, 1]


def compute_diag_ATA(stencils, num_voxels):
    indices, weights = stencils  # both [N_particles, K]
    weights_squared = weights ** 2

    diag = torch.zeros(num_voxels, device=weights.device)
    for k in range(weights.shape[1]):
        idx = indices[:, k]
        w2 = weights_squared[:, k]
        valid = (idx >= 0) & (idx < num_voxels)
        idx = idx.clone()
        idx[~valid] = 0  # dummy index
        w2[~valid] = 0
        diag.scatter_add_(0, idx, w2)
    return diag


def build_diag_preconditioner(stencil_weights, stencil_indices, n_cells):
    """
    Build diagonal preconditioner M⁻¹ ≈ A⁻¹ from valid stencil weights.

    Parameters
    ----------
    stencil_weights : (n_obs, N, K)
    stencil_indices : (n_obs, N, K)
    n_cells : int

    Returns
    -------
    M_inv : Callable(x) → x / diag
    """
    device = stencil_weights.device
    weights_squared = stencil_weights ** 2               # (n_obs, N, K)

    # Build validity mask for stencil indices
    valid = (stencil_indices >= 0) & (stencil_indices < n_cells)  # same shape

    # Zero-out invalid contributions
    valid_weights_squared = weights_squared * valid       # (n_obs, N, K)

    # Sum over all obs and stencil terms
    A_diag = valid_weights_squared.sum(dim=1)         # (N,)

    A_diag = A_diag.clamp_min(1e-8)  # avoid divide-by-zero


    return A_diag


def create_diagnostic_image(m_source, omega_source, device, dtype):
    """
    Create a diagnostic image that makes coordinate mapping problems obvious.

    Returns a combination of:
    1. Grid pattern - shows folding/distortion
    2. Coordinate encoding - shows exact positions
    3. Checkerboard - shows interpolation issues
    4. Radial pattern - shows rotational effects
    """

    # Get 2D grid for the slice
    H, W = m_source[-2:]
    y_coords = torch.linspace(omega_source[-4], omega_source[-3], H,
                              device=device, dtype=dtype)
    x_coords = torch.linspace(omega_source[-2], omega_source[-1], W,
                              device=device, dtype=dtype)

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # 1. GRID PATTERN - shows geometric distortion clearly
    grid_spacing = 8  # pixels
    grid_pattern = (
            (torch.sin(
                2 * np.pi * (torch.arange(H, device=device) / grid_spacing))[
             None, :].expand(H, W)) *
            (torch.sin(
                2 * np.pi * (torch.arange(W, device=device) / grid_spacing))[:,
             None].expand(H, W).T)
    )

    # 2. COORDINATE ENCODING - encode actual coordinates as intensity
    # This lets you see exactly where each pixel maps to
    coord_encoded = (xx - omega_source[-2]) / (
                omega_source[-1] - omega_source[-2])  # Normalize x to [0,1]
    coord_encoded += 2 * (yy - omega_source[-4]) / (
                omega_source[-3] - omega_source[-4])  # Add y*2 to [0,3]

    # 3. CHECKERBOARD - reveals interpolation artifacts
    checkerboard = ((torch.arange(H, device=device)[:, None] // 4) +
                    (torch.arange(W, device=device)[None, :] // 4)) % 2

    # 4. RADIAL PATTERN - shows rotational effects
    center_y, center_x = H // 2, W // 2
    dy = torch.arange(H, device=device, dtype=dtype)[:, None] - center_y
    dx = torch.arange(W, device=device, dtype=dtype)[None, :] - center_x
    radius = torch.sqrt(dy ** 2 + dx ** 2)
    radial = torch.sin(2 * np.pi * radius / 10)

    # 5. CORNER MARKERS - bright spots at corners to track them
    corner_markers = torch.zeros_like(xx)
    corner_size = 3
    corner_markers[:corner_size, :corner_size] = 1.0  # Top-left
    corner_markers[:corner_size, -corner_size:] = 2.0  # Top-right
    corner_markers[-corner_size:, :corner_size] = 3.0  # Bottom-left
    corner_markers[-corner_size:, -corner_size:] = 4.0  # Bottom-right

    # Combine all patterns
    diagnostic = (
            0.0 * grid_pattern +  # Grid shows geometric distortion
            0.0 * coord_encoded +  # Coordinates show exact mapping
            0.3 * checkerboard +  # Checkerboard shows interpolation
            0.0 * radial +  # Radial shows rotation effects
            0.0 * corner_markers  # Corners for reference
    )

    # Normalize to reasonable range
    diagnostic = (diagnostic - diagnostic.min()) / (
                diagnostic.max() - diagnostic.min())

    return diagnostic






# def build_pic_stencils_2d_with_j(omega, m_source, m_target, xp, do_derivative=False, return_jacobian=False):
#     """
#     Build interpolation stencils for matrix-free PIC push-forward operator.
#
#     Parameters
#     ----------
#     omega : list or torch.Tensor of shape (4,)
#         Domain bounds [x0, x1, y0, y1].
#     m_target : tuple (H, W)
#         Target grid size.
#     xp : torch.Tensor of shape (2, N)
#         Particle positions in physical space.
#     pwidth : int
#         Half-width of interpolation stencil.
#
#     Returns
#     -------
#     indices : torch.LongTensor [N, (2p+1)^2]
#         Flattened voxel indices per particle.
#     weights : torch.FloatTensor [N, (2p+1)^2]
#         Corresponding interpolation weights.
#     dT : Callable (rho: [num_voxels]) → torch.sparse.Tensor [N, num_voxels, 2], optional
#         Only returned if `do_derivative=True`.
#         A callable that, for a given input image `rho`, returns the derivative of
#         the interpolation operator with respect to particle positions (∂x, ∂y).
#     Jac : torch.FloatTensor [N], optional
#         Only returned if `return_jacobian=True`.
#         Sum of interpolation weights per particle, representing the Jacobian
#         determinant (mass-preserving factor) under the current deformation.
#     """
#     device = xp.device
#
#     # source grid settings
#     n_cells = torch.prod(m_source)
#     cell_size = (omega[1::2] - omega[0::2]) / m_source
#
#     # target grid settings
#     n_particles = torch.prod(m_target)
#     particle_size = (omega[1::2] - omega[0::2]) / m_target
#
#     x_vox = (xp[0] - omega[0]) / particle_size[0]
#     y_vox = (xp[1] - omega[2]) / particle_size[1]
#
#     x_vox = x_vox
#     y_vox = y_vox
#
#     x_vox = x_vox.permute(1, 0)
#     y_vox = y_vox.permute(1, 0)
#
#     Px = torch.floor(x_vox).long()
#     wx = x_vox - (Px.float())
#     Py = torch.floor(y_vox).long()
#     wy = y_vox - (Py.float())
#
#     Px = Px.reshape(-1)
#     Py = Py.reshape(-1)
#     wx = wx.reshape(-1)
#     wy = wy.reshape(-1)
#
#     epsP = torch.tensor(1*cell_size, device=device, dtype=xp.dtype)
#     # epsP = 0.1*cell_size
#     pwidth = torch.ceil(epsP / cell_size).to(torch.int32)
#     # pwidth = [3,3]
#
#     # # # # # Evaluate 1D basis
#     Bx, Dx = int1DSingle(wx, pwidth[0], epsP[0], cell_size[0],
#                          particle_size[0], do_derivative=True)  # [2*p+1, N]
#     By, Dy = int1DSingle(wy, pwidth[1], epsP[1], cell_size[1],
#                          particle_size[1], do_derivative=True)
#
#     # Bx = Bx / Bx.sum(0)
#     # By = By / By.sum(0)
#
#     nbx = Bx.shape[0]
#     nby = By.shape[0]
#     nVoxel = nbx * nby
#
#     I = torch.empty(nVoxel * n_cells, dtype=torch.long, device=device)
#     J = torch.empty(nVoxel * n_cells, dtype=torch.long, device=device)
#     B = torch.empty(nVoxel * n_cells, dtype=xp.dtype, device=device)
#
#     if do_derivative:
#         dBx = torch.empty_like(B)
#         dBy = torch.empty_like(B)
#
#     pp = 0
#     for i, px in enumerate(range(-pwidth[0], pwidth[0] + 1)):
#         for j, py in enumerate(range(-pwidth[1], pwidth[1] + 1)):
#             idx = slice(pp * n_cells, (pp + 1) * n_cells)
#             pp += 1
#
#             x_idx = Px + px
#             y_idx = Py + py
#             Iij = y_idx * m_target[1] + x_idx
#             Bij = Bx[i, :] * By[j, :]  # Elementwise per-particle weight
#
#             I[idx] = Iij
#             J[idx] = torch.arange(n_cells, device=Px.device)
#             B[idx] = Bij
#
#             if do_derivative:
#                 dBx[idx] = Dx[i, :] * By[j, :]
#                 dBy[idx] = Bx[i, :] * Dy[j, :]
#
#     invalid_idx = (I < 0) | (I >= n_particles)
#     valid_idx = ~invalid_idx
#     I[invalid_idx] = 0
#     J[invalid_idx] = 0
#     B[invalid_idx] = 0
#
#     stencil_size = nbx * nby
#
#     indices = I.view(stencil_size, n_cells)
#     weights = B.view(stencil_size, n_cells)
#     indices = indices.permute(1, 0)
#     weights = weights.permute(1, 0)
#     jvals = J.view(stencil_size, n_cells)
#     jvals = jvals.permute(1, 0)
#     valid_idx_mask = valid_idx.view(stencil_size, n_cells)
#     valid_idx_mask = valid_idx_mask.permute(1, 0)
#
#     results = [indices, weights, jvals]
#
#     if do_derivative:
#         dBx[invalid_idx] = 0
#         dBy[invalid_idx] = 0
#
#         dBx = dBx.reshape(stencil_size, n_cells)
#         dBy = dBy.reshape(stencil_size, n_cells)
#         dBx = dBx.permute(1, 0)
#         dBy = dBy.permute(1, 0)
#         # dBx = dBx.view(-1)
#         # dBy = dBy.view(-1)
#
#
#
#         def apply_dT_T(residual):
#             # Inputs: residual [n_cells, B] or [n_cells]
#             if residual.ndim == 1:
#                 residual = residual.unsqueeze(1)  # [n_cells, 1]
#
#             vals_x = residual[indices].squeeze(-1) * dBx # [K*N, B]
#             vals_y = residual[indices].squeeze(-1) * dBy
#
#             vals_x = (residual[indices].squeeze(-1) * dBx).sum(dim=1)
#             vals_y = (residual[indices].squeeze(-1) * dBy).sum(dim=1)
#
#             # vals_x = (residual[indices] * dBx).sum(dim=1)
#             # vals_y = (residual[indices] * dBy).sum(dim=1)
#
#             g = torch.zeros((n_cells, 2), device=residual.device)
#             g[:, 0].scatter_add_(0,
#                                  torch.arange(n_cells, device=residual.device),
#                                  vals_x / cell_size[0])
#             g[:, 1].scatter_add_(0,
#                                  torch.arange(n_cells, device=residual.device),
#                                  vals_y / cell_size[1])
#             return g.squeeze(1) # Sum over x/y for Gauss-Newton
#         results.append(apply_dT_T)
#
#     Jac = torch.zeros(n_cells, device=device)
#     Jac = Jac.scatter_add(0, J[~invalid_idx], B[~invalid_idx])
#     # B = B / (Jac[J] + 1e-6)
#     if return_jacobian:
#         results.append(Jac)
#
#     return tuple(results) if len(results) > 1 else results[0]
