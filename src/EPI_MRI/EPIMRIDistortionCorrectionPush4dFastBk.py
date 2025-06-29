import torch
from EPI_MRI.ImageModels import *
from EPI_MRI.InitializationMethods import *
from EPI_MRI.Preconditioners import *
import torchsparsegradutils as tsgu
#import jax
from torchsparsegradutils.jax import sparse_solve_j4t
#from torchsparsegradutils.cupy import sparse_solve_c4t
from EPI_MRI.ParticleInCell2D import *
from optimization.LinearSolvers import *


class EPIMRIDistortionCorrectionPush4dFastBk:
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
                 target_res=None,
                 averaging_operator=myAvg1D,
                 derivative_operator=myDiff1D,
                 regularizer=myLaplacian3D,
                 rho=0.0,
                 lambda_smooth=0.01,
                 initialization=InitializeCF,
                 PC=JacobiCG):
        self.dataObj = data_obj
        self.device = data_obj.device
        self.dtype = data_obj.dtype
        self.recon_image = None

        self.m_source = self.dataObj.m
        self.omega_source = self.dataObj.omega
        self.h_source = self.dataObj.h

        if target_res is None:
            self.m_target = self.dataObj.m
        else:
            self.m_target = torch.tensor([*self.m_source[:-2], *target_res],
                                         dtype=torch.int32,
                                         device=self.device)
            self.omega_target = self.omega_source
            self.h_target = self.h_source / self.m_target * self.m_source

        self.A = averaging_operator(self.omega_source[2:], self.m_source[1:],
                                    self.dtype, self.device)
        self.D_slice = derivative_operator(self.omega_source[-4:],
                                           self.m_source[-2:], self.dtype,
                                           self.device)
        self.D_image = derivative_operator(self.omega_source[-4:],
                                           self.m_target[-2:], self.dtype,
                                           self.device)
        self.D = derivative_operator(self.omega_source[-6:], self.m_source[-3:],
                                     self.dtype, self.device)
        self.xc = get_cell_centered_grid(self.omega_source, self.m_source,
                                         device=self.device,
                                         dtype=self.dtype).reshape(
            tuple(self.dataObj.m))
        self.S = QuadRegularizer(
            regularizer(self.omega_source[2:], self.dataObj.m[1:], self.dtype,
                        self.device))
        self.Q = TikRegularizer(self.omega_source, self.m_source)
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
        self.solver = PCG(max_iter=40, tol=1e-3, verbose=False)
        # --- NEW: Get full coordinate grid (all dims) ---
        self.base_grid = get_cell_centered_grid(self.dataObj.omega[2:],
                                                self.dataObj.m[1:],
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
        self.base_grid = get_cell_centered_grid(self.omega_source[2:],
                                                self.m_source[1:],
                                                device=self.device,
                                                dtype=self.dataObj.dtype,
                                                return_all=True)
        self.base_grid = self.base_grid.reshape(3, -1)

        self.n_particles = torch.prod(self.m_source[2:]).long()
        self.n_cells = torch.prod(self.m_target[2:]).long()
        self.v_pe = torch.tensor([0.0, 0.0, 1.0], device=self.device,
                                 dtype=self.dataObj.dtype)
        self.image_center = 0.5 * (
                torch.tensor(self.dataObj.omega[3::2]) + torch.tensor(
            self.dataObj.omega[2::2]))  # (x_c, y_c, z_c)

        self.Lx, self.Ly = get_2d_gradient_matrix(
            self.m_target[-2],
            self.m_target[-1],
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

        Dc = 0.5 * (residuals ** 2 * jlsq[:, None]).mean()
        grad = (dlsq * jlsq[:, None]).mean(dim=(0, 1))
        dD = self.D.transp_mat_mul(grad)

        # compute distance measure
        hd_source = torch.prod(self.h_source)
        hd_target = torch.prod(self.h_target)

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

        save_data(dD.permute(1, 2, 0),
                  f"/home/laurin/workspace/PyHySCO/data/results/debug/dD.nii.gz")
        save_data(dS.permute(1, 2, 0),
                  f"/home/laurin/workspace/PyHySCO/data/results/debug/dS.nii.gz")
        save_data(dP.permute(1, 2, 0),
                  f"/home/laurin/workspace/PyHySCO/data/results/debug/dP.nii.gz")
        save_data(grad.permute(1, 2, 0),
                  f"/home/laurin/workspace/PyHySCO/data/results/debug/grad.nii.gz")
        save_data(geom.permute(1, 2, 0),
                  f"/home/laurin/workspace/PyHySCO/data/results/debug/geom.nii.gz")
        save_data(intensity.permute(1, 2, 0),
                  f"/home/laurin/workspace/PyHySCO/data/results/debug/intensity.nii.gz")


        if not calc_hessian:
            return Jc, dJ
        else:
            def H(x):
                """ Matrix-vector product between Hessian and a tensor x of size m_plus(m). """
                Dx = self.D.mat_mul(x)
                dr = geom * Dx + intensity * self.A.mat_mul(x)
                dr_d2psi = dr * hd_source
                if self.beta == 0:  # d2P is zeros
                    d2D = self.D.transp_mat_mul(
                        dr_d2psi * geom) + self.A.transp_mat_mul(
                        dr_d2psi * intensity)
                    return d2D + hd_source * self.alpha * d2S.mat_mul(
                        x) + hd_source * self.rho * x
                else:
                    d2D = self.D.transp_mat_mul(
                        dr_d2psi * geom + hd_source * self.beta * d2G * Dx) + self.A.transp_mat_mul(
                        dr_d2psi * intensity)
                    return d2D + hd_source * self.alpha * d2S.mat_mul(
                        x) + hd_source * self.rho * x

            if self.PC is not None:
                diagD, diagP, diagS = self.PC.getM(geom, intensity, hd_source, d2G,
                                                   self.D,
                                                   self.A, self.S.H, self.alpha,
                                                   self.beta)
                self.PC.M += hd_source * self.rho
                M = lambda x: self.PC.eval(x)
            else:
                M = None

            # M = None
            #
            # # Even more simplified version focusing only on geometric term:
            # def H_geometric_only(x):
            #     """
            #     Alternative: Focus only on geometric deformation effects.
            #     This might be more appropriate for EPI correction.
            #     """
            #     Dx = self.D.mat_mul(x)
            #     dr_geom = geom * Dx * hd_target
            #
            #     # Only geometric coupling
            #     d2D_geom = self.D.transp_mat_mul(dr_geom * geom)
            #
            #     # Add regularization
            #     return d2D_geom + hd_source * self.alpha * d2S.mat_mul(
            #         x) + self.rho * x
            #
            # # assert isinstance(D, Conv1D)
            # #
            # # For geometric-only case, we only need D^T D term
            # D2 = self.D.op_mul(self.D)
            #
            # # Geometric data fidelity term: only geom^2 * D^T D
            # diagD = D2.transp_mat_mul(geom ** 2)
            #
            # # Smoothness term (unchanged)
            # diagS = self.S.H.diag()
            #
            # # Build preconditioner (no intensity or beta terms)
            # M = hd_source * diagD + hd_source * self.alpha * diagS

            return Jc, dJ, H, M




    def solve_lsq(self, yc):
        bc = self.A.mat_mul(yc).reshape(-1,1)  # averaging matrix & translation vector
        bc_3d = (bc * self.v_pe.view(1, -1)).T  # shift vector in original space
        self.bc = bc

        recon_vols = []
        jac = torch.zeros((self.images.shape[0], *self.m_source[1:]), dtype=self.dtype, device=self.device)
        jacobians = torch.zeros(*self.images.shape, dtype=self.dtype, device=self.device)
        residuals = torch.zeros(*self.images.shape, dtype=self.dtype, device=self.device)

        for slice_index in range(self.images.shape[2]):

            stencil_weights = []
            stencil_indices = []
            d_images = []
            diag = torch.zeros(self.n_cells, device=self.device)

            for pe_dir_index in range(self.images.shape[0]):
                xp = self.base_grid + self.pe_dirs[
                    pe_dir_index] * bc_3d
                rot_mat_permuted = self.rel_mats[pe_dir_index, :3, :3]
                xp = rot_mat_permuted @ (xp - self.image_center.unsqueeze(
                    1)) + self.image_center.view(3, 1)
                xp = xp[1:, :].reshape(2, *self.m_source[1:])

                indices, weights, d_image, jac_slice = build_pic_stencils_2d(
                    self.omega_source[-4:],
                    self.m_source[-2:],
                    self.m_target[-2:],
                    xp[:, slice_index, :, :],
                    do_derivative=True,
                    return_jacobian=True)

                jac[pe_dir_index,slice_index] += jac_slice.reshape(*self.m_source[2:])

                stencil_indices.append(indices)
                stencil_weights.append(weights)
                d_images.append(d_image)

                diag += compute_diag_ATA((indices, weights), self.n_cells)

            stencil_indices = torch.stack(stencil_indices, dim=0)
            stencil_weights = torch.stack(stencil_weights, dim=0)

            recon_vol_slices = []
            for vol_index in range(self.images.shape[1]):

                recon_vol_slice, res, dC = self.solve_lsq_slice(vol_index, slice_index, stencil_indices, stencil_weights, d_images, diag)
                recon_vol_slices.append(recon_vol_slice)
                residuals[:,vol_index,slice_index] = res.reshape(-1, *self.m_source[2:])
                jacobians[:,vol_index,slice_index] = dC.reshape(-1, *self.m_source[2:])

            recon_vols.append(torch.stack(recon_vol_slices, dim=0))

        recon_image = torch.stack(recon_vols, dim=0)
        recon_image = recon_image.reshape(*recon_image.shape[0:2], *self.m_target[2:])
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

        stencil_indices_masked = stencil_indices.view(-1, stencil_indices.shape[-1])[pe_slice_masks_flat]
        stencil_weights_masked = stencil_weights.view(-1, stencil_indices.shape[-1])[pe_slice_masks_flat]
        pe_slices_masked = pe_slices[pe_slice_masks]

        def A_fn(x):
            x_gathered = x[stencil_indices_masked]  # [N, K, 1]
            Ax = (x_gathered * stencil_weights_masked.unsqueeze(-1)).sum(dim=1)  # [N, 1]

            # Expand Ax to [N, K] to match weights
            Ax_expanded = Ax.expand(-1, stencil_weights_masked.shape[1])  # [N, K]
            contrib = Ax_expanded * stencil_weights_masked  # [N, K]

            # Flatten everything
            flat_indices = stencil_indices_masked.reshape(-1)  # [N*K]
            flat_contrib = contrib.reshape(-1, 1)  # [N*K, 1]

            # Scatter-add all at once
            result = torch.zeros((self.n_cells, 1), device=x.device)
            result.scatter_add_(0, flat_indices.unsqueeze(1), flat_contrib)

            # Regularization: lambda * (Lxᵗ Lx + Lyᵗ Ly) x
            if self.lambda_smooth > 0:
                LTLx = self.LxT_Lx @ x + self.LyT_Ly @ x
                result += self.lambda_smooth * LTLx

            return result

        pe_masked = pe_slices * pe_slice_masks
        contrib = pe_slices_masked.unsqueeze(1) * stencil_weights_masked  # [K*N, S]
        flat_indices = stencil_indices_masked.reshape(-1)  # [K*N*S]
        flat_contrib = contrib.reshape(-1, 1)  # [K*N*S, 1]

        # Scatter-add to get RHS
        rhs = torch.zeros((self.n_cells, 1), device=self.device)
        rhs.scatter_add_(0, flat_indices.unsqueeze(1), flat_contrib)

        if self.recon_image is None:
            x0 = torch.zeros_like(rhs)
        else:
            x0 = self.recon_image[vol_index,slice_index].view(-1, 1)

        def make_diag_preconditioner(diag):
            diag = diag.view(-1, 1)  # ensure it's column vector

            def M(x):
                x = x.view(-1, 1)  # ensure x is also column vector
                return x / (diag + 1e-6)  # safe: [N, 1] / [N, 1]

            return M


        M = make_diag_preconditioner(diag)
        rhocorr, _, _, _, _ = self.solver.eval(A=A_fn, b=rhs, M=M, x=x0)

        gathered = rhocorr[stencil_indices].squeeze(-1)  # [N_particles, K, B]
        weighted = gathered * stencil_weights # [N_particles, K, B]
        interpolated = weighted.sum(dim=-1)  # [N_particles, B]
        residuals = interpolated-pe_slices

        # save_data(residuals.reshape(-1, *self.m_source[2:]).permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/residuals.nii.gz")
        # save_data(interpolated.reshape(-1, *self.m_source[2:]).permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/interpolated.nii.gz")
        # save_data(pe_slices.reshape(-1, *self.m_source[2:]).permute(1, 2, 0),
        #           f"/home/laurin/workspace/PyHySCO/data/results/debug/obs.nii.gz")

        dT_imgs = []
        for pe_index in range(pe_slices.shape[0]):
            d = derivative_ops[pe_index](rhocorr)

            dx_weighted = residuals[pe_index]*d[:,0]
            dy_weighted = residuals[pe_index]*d[:,1]

            R = self.rel_mats[pe_index][:3, :3]
            v_rot = R @ self.v_pe
            v_inplane = v_rot[1:]
            v_inplane = v_inplane / torch.norm(v_inplane)
            dy_contrib = dx_weighted * v_inplane[0] + dy_weighted * v_inplane[1]
            dT_imgs.append(dy_contrib.reshape(*self.m_source[2:]))
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

    x_vox = x_vox.permute(1, 0)
    y_vox = y_vox.permute(1, 0)

    Px = torch.ceil(x_vox).long()
    wx = x_vox - (Px.float() - 0)
    Py = torch.ceil(y_vox).long()
    wy = y_vox - (Py.float() - 0)

    Px = Px.reshape(-1)
    Py = Py.reshape(-1)
    wx = wx.reshape(-1)
    wy = wy.reshape(-1)

    pwidth = torch.ceil(particle_size / cell_size).to(torch.int32)
    # pwidth = [4,4]

    # # # # # Evaluate 1D basis
    Bx, Dx = int1DSingle(wx, pwidth[0], particle_size[0], cell_size[0],
                         particle_size[0], do_derivative=True)  # [2*p+1, N]
    By, Dy = int1DSingle(wy, pwidth[1], particle_size[1], cell_size[1],
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
            Iij = y_idx * m_target[1] + x_idx
            Bij = Bx[i, :] * By[j, :]  # Elementwise per-particle weight

            I[idx] = Iij
            J[idx] = torch.arange(n_cells, device=Px.device)
            B[idx] = Bij

            if do_derivative:
                dBx[idx] = Dx[i, :] * By[j, :]
                dBy[idx] = Bx[i, :] * Dy[j, :]

    invalid_idx = (I < 0) | (I >= n_particles)
    I[invalid_idx] = -1
    J[invalid_idx] = -1
    B[invalid_idx] = 0

    stencil_size = nbx * nby

    indices = I.view(stencil_size, n_cells)
    weights = B.view(stencil_size, n_cells)
    indices = indices.permute(1, 0)
    weights = weights.permute(1, 0)

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
