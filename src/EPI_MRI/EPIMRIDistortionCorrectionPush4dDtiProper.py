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


class EPIMRIDistortionCorrectionPush4dDtiProper:
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
        self.solver = PCG(max_iter=40, tol=1e-3, verbose=False)
        # --- NEW: Get full coordinate grid (all dims) ---
        self.base_grid = get_cell_centered_grid(self.omega_recon,
                                                self.m_recon,
                                                device=self.device,
                                                dtype=self.dataObj.dtype,
                                                return_all=True)

        images = []
        mean_b0_images = []
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

            pe_b0_idx = pe_bval < 50
            rpe_b0_idx = rpe_bval < 50

            pe_dti_idx = pe_bval > 900
            rpe_dti_idx = rpe_bval > 900

            pe_b0_volumes = pe_image.data[pe_b0_idx]
            rpe_b0_volumes = pe_image.data[rpe_b0_idx]
            pe_b0_mean = torch.mean(pe_b0_volumes, dim=0)
            rpe_b0_mean = torch.mean(rpe_b0_volumes, dim=0)

            pe_dti_volumes = pe_image.data[pe_dti_idx]
            rpe_dti_volumes = rpe_image.data[rpe_dti_idx]

            mean_b0_images.append(pe_b0_mean.repeat(pe_dti_volumes.shape[0], 1, 1, 1))
            mean_b0_images.append(rpe_b0_mean.repeat(rpe_dti_volumes.shape[0], 1, 1, 1))

            pe_bvec = pe_bvec[:,pe_dti_idx]
            rpe_bvec = rpe_bvec[:,rpe_dti_idx]
            pe_bval = pe_bval[pe_dti_idx]
            rpe_bval = rpe_bval[rpe_dti_idx]

            pe_dti_volumes = pe_dti_volumes / pe_b0_mean
            rpe_dti_volumes = rpe_dti_volumes / rpe_b0_mean

            pe_mask_repeated = pe_mask.unsqueeze(0).repeat(pe_dti_volumes.shape[0], 1, 1, 1)
            rpe_mask_repeated = pe_mask.unsqueeze(0).repeat(rpe_dti_volumes.shape[0], 1, 1, 1)

            masks.append(pe_mask_repeated)
            masks.append(rpe_mask_repeated)


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

            rel_mat = torch.linalg.inv(self.dataObj.rel_mats[i])
            rel_mat = self.dataObj.rel_mats[i]

            rel_mats.append(rel_mat)
            # rel_mats.append(rel_mat)

        self.mean_b0_images = torch.stack(mean_b0_images, dim=0)
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
                torch.tensor(self.dataObj.omega[3::2]) + torch.tensor(
            self.dataObj.omega[2::2]))  # (x_c, y_c, z_c)

        self.Lx, self.Ly = get_2d_gradient_matrix(
            self.m_recon[-2],
            self.m_recon[-1],
            self.device,
            self.dtype)


        self.dti_images = torch.stack(images, dim=0)
        self.masks = torch.cat(masks, dim=0)
        self.bvec = torch.stack(bvec, dim=0)
        self.bval = torch.stack(bval, dim=0)
        self.rel_mats = torch.stack(rel_mats, dim=0)
        self.pe_dirs = torch.tensor(pe_dirs, dtype=self.dtype, device=self.device)
        self.particle_grid = get_cell_centered_grid(self.dataObj.omega[2:],
                                    self.m_recon[1:],
                                    device=self.device,
                                    dtype=self.dataObj.dtype,
                                    return_all=True)

        Lx, Ly = get_2d_gradient_matrix(self.m_recon[-2],
                                        self.m_recon[-1], self.device,
                                        self.dtype)

        I6 = torch.eye(6).to_sparse().to(device=self.device,
                                         dtype=self.dtype)

        self.Gx = sparse_kron(I6, Lx)  # [6 * N_x_edges, 6 * N_voxels]
        self.Gy = sparse_kron(I6, Ly)  # [6 * N_y_edges, 6 * N_voxels]

        self.LxT_Lx = compute_diag_LTL(self.Gx, self.Gy)

        self.GxT_Gx = self.Gx.T @ self.Gx
        self.GyT_Gy = self.Gy.T @ self.Gy



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

        Dc = (0.5 * (residuals ** 2 * jlsq[:, None])).mean(dim=[0,1]).sum()
        grad = (dlsq * jlsq[:, None]).mean(dim=(0, 1))

        pad_w = yc.shape[-1] - grad.shape[-1]  # Should be 1
        # Pad at the beginning instead of the end
        dD = torch.nn.functional.pad(grad, (1, 0),
                                     mode='replicate')  # Now (66,67)

        dD_pre = self.D.transp_mat_mul(grad)

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
        save_data(dD_pre.permute(1, 2, 0),
                  f"/home/laurin/workspace/PyHySCO/data/results/debug/dDDiff.nii.gz")
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
            # def H(x):
            #     """ Matrix-vector product between Hessian and a tensor x of size m_plus(m). """
            #     Dx = self.D.mat_mul(x)
            #     dr = geom * Dx + intensity * self.A.mat_mul(x)
            #     dr_d2psi = dr * hd_source
            #     if self.beta == 0:  # d2P is zeros
            #         d2D = self.D.transp_mat_mul(
            #             dr_d2psi * geom) + self.A.transp_mat_mul(
            #             dr_d2psi * intensity)
            #         return d2D + hd_source * self.alpha * d2S.mat_mul(
            #             x) + hd_source * self.rho * x
            #     else:
            #         d2D = self.D.transp_mat_mul(
            #             dr_d2psi * geom + hd_source * self.beta * d2G * Dx) + self.A.transp_mat_mul(
            #             dr_d2psi * intensity)
            #         return d2D + hd_source * self.alpha * d2S.mat_mul(
            #             x) + hd_source * self.rho * x
            #
            # if self.PC is not None:
            #     diagD, diagP, diagS = self.PC.getM(geom, intensity, hd_source, d2G,
            #                                        self.D,
            #                                        self.A, self.S.H, self.alpha,
            #                                        self.beta)
            #     self.PC.M += hd_source * self.rho
            #     M = lambda x: self.PC.eval(x)
            # else:
            #     M = None

            # # M = None
            # #
            # # Even more simplified version focusing only on geometric term:
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
        jacobians = torch.zeros(*self.images.shape, dtype=self.dtype, device=self.device)
        residuals = torch.zeros(*self.images.shape, dtype=self.dtype, device=self.device)

        lambda_smooth = 5000000  # or any other value you like
        # sqrt_lambda = torch.sqrt(
        #     torch.tensor(lambda_smooth, device=A.device, dtype=A.dtype))


        v = torch.tensor([0.0, 0.0, 1.0], device=self.device,
                         dtype=self.dataObj.dtype)
        bc1 = self.A.mat_mul(yc).reshape(-1,
                                         1)  # averaging matrix & translation vector
        bc1_full = bc1 * v.view(1, -1)  # shift vector in original space

        # Process each PE-RPE pair

        particle_grids = []

        for i, pair in enumerate(self.dataObj.image_pairs):
            xp1 = self.particle_grid.view(3, -1) + bc1_full.T
            xp2 = self.particle_grid.view(3, -1) - bc1_full.T

            # xp1 = self.particle_grid.view(3, -1)
            # xp2 = self.particle_grid.view(3, -1)

            T_permuted = self.rel_mats[i][:3, :3]
            T_permuted = torch.linalg.inv(
                self.rel_mats[i][:3, :3])

            center = 0.5 * (
                    torch.tensor(self.omega_recon[3::2]) + torch.tensor(
                self.omega_recon[2::2]))  # (x_c, y_c, z_c)

            xp1 = T_permuted @ (xp1 - center.unsqueeze(1)) + center.view(3, 1)
            xp2 = T_permuted @ (xp2 - center.unsqueeze(1)) + center.view(3, 1)

            xp1 = xp1[1:, :].reshape(2, *self.m_recon[1:])
            xp2 = xp2[1:, :].reshape(2, *self.m_recon[1:])

            particle_grids.append(
                xp1.repeat(self.dti_images.shape[1], 1, 1, 1, 1))
            particle_grids.append(
                xp2.repeat(self.dti_images.shape[1], 1, 1, 1, 1))

        particle_grids = torch.stack(particle_grids, dim=0)

        particle_grids_lin = particle_grids.view(-1, *particle_grids.shape[2:])
        dti_images_lin = self.dti_images.view(-1, *self.dti_images.shape[2:])
        mean_b0_lin = self.mean_b0_images.view(-1, *self.dti_images.shape[2:])
        bvals_lin = self.bval.view(-1)
        bvec_lin = self.bvec.view(-1, 3)
        # bvec_lin = bvec_lin.reshape(-1, *bvec_lin.shape[2:])
        masks_lin = self.masks.view(-1, *self.dti_images.shape[2:])
        slice_pic_matrices = []
        slice_pic_jacs = []
        slice_pic_dC = []

        Q = -torch.stack([
            bvec_lin[:, 0] ** 2,  # Dxx
            bvec_lin[:, 1] ** 2,  # Dyy
            bvec_lin[:, 2] ** 2,  # Dzz
            2 * bvec_lin[:, 0] * bvec_lin[:, 1],  # Dxy
            2 * bvec_lin[:, 0] * bvec_lin[:, 2],  # Dxz
            2 * bvec_lin[:, 1] * bvec_lin[:, 2]  # Dyz
        ], dim=1)
        Q = Q * bvals_lin.unsqueeze(1)

        rhocorr_slices = []
        fa_slices = []

        for slice_index in range(dti_images_lin.shape[1]):

            obs = []
            obs_masks = []
            mean_b0_imgs = []
            stencil_weights_vol = []
            stencil_indices_vol = []

            for image_index in range(dti_images_lin.shape[0]):
                indices, weights, d_image, jac_slice = build_pic_stencils_2d(self.omega_recon[-4:],
                                                 self.m_recon[-2:],
                                                 self.m_distorted[-2:],
                                                 particle_grids_lin[image_index,
                                                 :, slice_index].clone(),
                                                 do_derivative=True,
                                                 return_jacobian=True)
                stencil_indices_vol.append(indices)
                stencil_weights_vol.append(weights)

                obs.append(dti_images_lin[image_index, slice_index].view(-1))
                obs_masks.append(masks_lin[image_index, slice_index].view(-1))
                mean_b0_imgs.append(mean_b0_lin[image_index, slice_index].view(-1))
            # obs.append(dti_images_lin[image_index, slice_index])

            stencil_indices_vol = torch.stack(stencil_indices_vol, dim=0)
            stencil_weights_vol = torch.stack(stencil_weights_vol, dim=0)

            obs_image = torch.stack(obs, dim=0)
            save_data(obs_image.reshape(-1, *self.m_distorted[2:]),
                      "/home/laurin/workspace/PyHySCO/data/results/debug/obs.nii.gz")
            obs_image = torch.log(obs_image)
            save_data(obs_image.reshape(-1, *self.m_distorted[2:]),
                      "/home/laurin/workspace/PyHySCO/data/results/debug/obs_log.nii.gz")

            obs_masks = torch.stack(obs_masks, dim=0)
            mean_b0_imgs = torch.stack(mean_b0_imgs, dim=0)

            valid_mask = obs_masks & torch.isfinite(
                obs_image)
            obs_image[~valid_mask] = 0

            roi_stencils = []
            for k in range(stencil_indices_vol.shape[0]):
                stencil = (stencil_indices_vol[k, :, :],
                           stencil_weights_vol[k, :, :])
                roi_stencils.append(stencil)

            # def A_fn(x):
            # 	result = torch.zeros_like(x)
            #
            # 	xres = x.reshape(-1,6)
            # 	for i in range(Q.shape[1]):
            # 		Q_i = Q[i]  # shape (6,)
            # 		stencil = roi_stencils[
            # 			i]  # particle-to-voxel stencil for this DWI
            #
            # 		indices, weights = stencil
            # 		weights = weights.clone()
            # 		# weights[~valid_mask[i], :] = 0
            #
            # 		stencil_filtered = (indices, weights)
            # 		AtAx_i = apply_pic_tensor_rmatvec(stencil_filtered, xres[:,i].unsqueeze(1),
            # 										  Q_i, self.m_recon[-2] *
            # 										  self.m_recon[-1])
            # 		Axi = apply_pic_tensor_matvec(stencil_filtered, AtAx_i,
            # 									   Q_i)  # A_i x with Q weighting
            #
            # 		result += Axi
            #
            # 	# Add smoothness regularization
            # 	if self.lambda_smooth > 0:
            # 		LTLx = self.GxT_Gx @ x + self.GyT_Gy @ x
            # 		result += self.lambda_smooth * LTLx
            #
            # 	return result





            def build_rhs(obs):
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

                # indices, weights = roi_stencils
                # N, K = weights.shape
                # n_channels = obs.shape[1]

                n_obs = obs.shape[0]
                device = obs.device

                # ---------------------------------------------------------------
                # loop over observations; K is tiny so this is cheap
                # ---------------------------------------------------------------

                b_is = []
                for i in range(0,6):
                    b_i = torch.zeros((self.n_particles, 1), device=device)
                    b0_weights_vol = b0_weights[i].unsqueeze(1)
                    for k in range(n_obs):
                        idx_k, w_k =  roi_stencils[k]

                        # gather grid values that stencil k references
                        y_k = obs[k]  # (n_cells, 1) or (n_cells,)
                        y_gathered = y_k[idx_k]  # (N, K, 1) or (N, K)

                        # adjoint interpolation: sum_j  w_ij · y(idx_ij)
                        if y_gathered.dim() == 3:  # (N,K,1) → (N,K)
                            y_gathered = y_gathered.squeeze(-1)

                        b_k = (y_gathered * (b0_weights_vol* w_k * Q[k,i])).sum(dim=1, keepdim=True)  # (N, 1)
                        b_i += b_k  # accumulate

                    b_is.append(b_i)

                b = torch.cat(b_is, dim=0)

                return b

            def A_fn(x):
                result = torch.zeros_like(x)

                # xres = x.reshape(6,-1)
                for i in range(Q.shape[1]):
                    Q_i = Q[i]  # shape (6,)
                    stencil = roi_stencils[
                        i]  # particle-to-voxel stencil for this DWI
                    b0_weights_vol = b0_weights[i].repeat(6).unsqueeze(1)

                    indices, weights = stencil
                    weights = weights.clone()
                    # weights[~valid_mask[i], :] = 0

                    stencil_filtered = (indices, weights)

                    AtAx_i = apply_pic_tensor_rmatvec(stencil_filtered, x,
                                                      Q_i, self.n_cells)

                    # save_data(AtAx_i.reshape(6, *self.m_distorted[2:]),
                    # 		  "/home/laurin/workspace/PyHySCO/data/results/debug/AtAx_i.nii.gz")
                    # save_data(x.reshape(6, *self.m_recon[2:]),
                    # 		  "/home/laurin/workspace/PyHySCO/data/results/debug/x_i.nii.gz")

                    AtAx_i = AtAx_i * b0_weights_vol

                    Axi = apply_pic_tensor_matvec(stencil_filtered, AtAx_i,
                                                  Q_i)  # A_i x with Q weighting

                    # save_data(Axi.reshape(6, *self.m_recon[2:]),
                    # 		  "/home/laurin/workspace/PyHySCO/data/results/debug/Axi.nii.gz")

                    # valid_idx = torch.isfinite(Axi)
                    # result[valid_idx] += Axi[valid_idx]
                    result += Axi

                save_data(result.reshape(6, *self.m_recon[2:]),
                          "/home/laurin/workspace/PyHySCO/data/results/debug/Axi_summed.nii.gz")

                # Add smoothness regularization
                if self.lambda_smooth > 0:
                    LTLx = self.GxT_Gx @ x + self.GyT_Gy @ x
                    result += self.lambda_smooth * LTLx

                # result[~torch.isfinite(result)] = 1e-12

                return result


            b0_weights = mean_b0_imgs/1000
            b0_weights = b0_weights**2
            save_data(weights.reshape(-1, *self.m_distorted[2:]).permute(1, 2, 0),
                      "/home/laurin/workspace/PyHySCO/data/results/debug/weights.nii.gz")

            rhs = build_rhs(obs_image)
            # rhs_mean_b0 = build_rhs_b0(mean_b0_imgs)
            # rhs_mean_b0 = rhs_mean_b0/10000
            # rhs_weighted = rhs * rhs_mean_b0
            # weights = rhs_mean_b0[0:self.n_particles]

            save_data(rhs.reshape(6, *self.m_recon[2:]).permute(1, 2, 0),
                      "/home/laurin/workspace/PyHySCO/data/results/debug/rhs_dti.nii.gz")
            # # save_data(mean_b0_imgs.reshape(6, *self.m_distorted[2:]).permute(1, 2, 0),
            # # 		  "/home/laurin/workspace/PyHySCO/data/results/debug/org_b0.nii.gz")
            # save_data(rhs_mean_b0.reshape(6, *self.m_recon[2:]).permute(1, 2, 0),
            # 		  "/home/laurin/workspace/PyHySCO/data/results/debug/rhs_b0.nii.gz")
            # save_data(rhs_weighted.reshape(6, *self.m_recon[2:]).permute(1, 2, 0),
            # 		  "/home/laurin/workspace/PyHySCO/data/results/debug/rhs_weighted.nii.gz")

            # save_data(rhs.reshape(*self.m_recon[2:], 6),
            # 		  "/home/laurin/workspace/PyHySCO/data/results/debug/rhs_dti.nii.gz")





            x = torch.randn_like(rhs)
            AtAx = A_fn(x)  # Aᵗ A x

            dot = torch.dot(x.flatten(), AtAx.flatten())
            print("xᵗ·AᵗAx =", dot.item())

            # rhs = torch.zeros(
            # 	(6 * self.m_recon[-2] * self.m_recon[-1], 1),
            # 	device=self.device)
            #
            # for i in range(obs_image.shape[0]):
            # 	# for k in range(0,6)
            # 	Q_i = Q[i]
            # 	stencil = roi_stencils[i]
            # 	b_i = obs_image[i].clone().unsqueeze(1)  # [N, 1]
            #
            # 	indices, weights = stencil
            # 	weights = weights.clone()
            # 	# weights[~valid_mask[i], :] = 0
            # 	stencil_filtered = (indices, weights)
            #
            # 	# b_i[~valid_mask[i]] = 1e-12
            #
            # 	# compute Aᵀb
            # 	Atb_i = apply_pic_tensor_rmatvec(
            # 		stencil_filtered, b_i, Q_i,
            # 		self.m_recon[-2] * self.m_recon[-1]
            # 	)
            # 	rhs += Atb_i

            u = torch.randn_like(rhs)
            v = torch.randn_like(rhs)

            # sym_err = torch.dot(u.flatten(), A_fn(v).flatten()) - torch.dot(
            # 	v.flatten(), A_fn(u).flatten())
            # curv_u = torch.dot(u.flatten(), A_fn(u).flatten())



            x0 = torch.zeros_like(rhs)

            def make_diag_preconditioner(diag):
                diag = diag.view(-1, 1)
                diag = diag.clamp_min(1e-8)
                def M(x):
                    return x / diag

                return M

            diag_data = torch.zeros((6 * self.m_recon[-2] * self.m_recon[-1]),
                                    device=self.device)

            for i in range(dti_images_lin.shape[0]):
                Q_i = Q[i]  # shape (6,)
                stencil = roi_stencils[i]
                diag_data += compute_diag_ATA_tensor_weighted(stencil, Q_i,
															  b0_weights[i],
                                                     self.m_recon[-2] *
                                                     self.m_recon[-1])

            precond_diag = diag_data + self.lambda_smooth * self.LxT_Lx
            # precond_diag = diag_data

            M = make_diag_preconditioner(precond_diag)


            solver = PCG(max_iter=300, tol=1e-10, verbose=True)
            rhocorr, _, _, _, _ = solver.eval(A=A_fn, b=rhs, M=M, x=x0)

            # save_data(rhocorr.reshape(66, 66),
            #           "/home/laurin/workspace/PyHySCO/data/results/debug/rhocorr_dti_dwi.nii.gz")

            D_raw = rhocorr.reshape(6, self.m_recon[-2], self.m_recon[-1])
            # r = r.permute(2,0,1)
            save_data(D_raw.permute(2,0,1),"/home/laurin/workspace/PyHySCO/data/results/debug/D2.nii.gz")

            save_data(D_raw.permute(2, 0, 1),
                      "/home/laurin/workspace/PyHySCO/data/results/debug/D2.nii.gz")


            D = D_raw.permute(1, 2, 0)

            fa = compute_fa_from_tensor(D)
            save_data(fa,
                      "/home/laurin/workspace/PyHySCO/data/results/debug/FA.nii.gz")

            rhocorr_slices.append(D)
            fa_slices.append(fa)

        rhocorr_image = torch.stack(rhocorr_slices, dim=0)

        save_data(rhocorr_image.permute(*self.dataObj.permute_back[0][1:], 3),
                  "/home/laurin/workspace/PyHySCO/data/results/debug/D.nii.gz")
        fa_image = torch.stack(fa_slices, dim=0)
        save_data(fa_image.permute(*self.dataObj.permute_back[0][1:]),
                  "/home/laurin/workspace/PyHySCO/data/results/debug/FA.nii.gz")



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

    epsP = torch.tensor(1*cell_size, device=device, dtype=xp.dtype)
    # epsP = 2*cell_size
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
            Iij = y_idx * m_target[1] + x_idx
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


# def apply_pic_tensor_matvec(stencils, x, Q):
# 	indices, weights = stencils  # [N, K]
# 	N, K = weights.shape
# 	num_vox = x.shape[0] // 6
#
# 	x = x.view(6, num_vox)  # [6, N_vox]
#
# 	# Output: r = sum_j w_ij * sum_k Q_k * x_k[indices_ij]
# 	r = torch.zeros((N,), device=x.device)
#
# 	for k in range(6):
# 		xk = x[k]  # shape [N_vox]
# 		r += Q[k] * apply_pic_matvec(stencils, xk.unsqueeze(1)).squeeze()
#
# 	return r.unsqueeze(1)  # shape [N, 1]

def apply_pic_tensor_matvec(stencils, x, Q):
    indices, weights = stencils  # [N, K]
    N, K = weights.shape
    num_vox = x.shape[0] // 6

    x = x.view(6,num_vox)  # [6, N_vox]

    # Output: r = sum_j w_ij * sum_k Q_k * x_k[indices_ij]
    r = torch.zeros((N,), device=x.device)
    r = []

    for k in range(6):
        xk = x[k,:]  # shape [N_vox]
        r.append(Q[k] * apply_pic_matvec(stencils, xk.unsqueeze(1)).squeeze())

    r = torch.cat(r, dim=0)

    return r.unsqueeze(1)  # shape [N, 1]



def apply_pic_tensor_rmatvec(stencils, r, Q, num_voxels):
    idx, w = stencils
    idx_flat = idx.view(-1, 1)
    N, K = w.shape
    n_channels = r.shape[1]

    assert n_channels == 1, "This version assumes r is (N, 1)"

    r_res = r.reshape(6,-1)
    out_imgs = []
    for i in range(6):  # 6 tensor components
        out = torch.zeros((num_voxels, 1), device=r.device)
        scaled = r_res[i,:].unsqueeze(1) * (w * Q[i])
        scaled_flat = scaled.view(-1,1)
        out.scatter_add_(0, idx_flat, scaled_flat)
        out_imgs.append(out)
    out = torch.cat(out_imgs)

    return out

# def apply_pic_tensor_rmatvec(stencils, r, Q, num_voxels):
# 	indices, weights = stencils
# 	N, K = weights.shape
# 	n_channels = r.shape[1]
#
# 	assert n_channels == 1, "This version assumes r is (N, 1)"
# 	out = torch.zeros((6 * num_voxels, 1), device=r.device)
#
# 	for k in range(K):
# 		idx = indices[:, k]
# 		mask = (idx >= 0) & (idx < num_voxels)
# 		safe_idx = idx.clone()
# 		safe_idx[~mask] = 0
# 		w = weights[:, k].clone()
# 		w[~mask] = 0
#
# 		for i in range(6):  # 6 tensor components
# 			scaled = r * (w * Q[i]).unsqueeze(1)
# 			out_idx = safe_idx + i * num_voxels
# 			out.scatter_add_(0, out_idx.unsqueeze(1), scaled)
# 			t = 1
#
# 	return out


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
    N, K = weights.shape
    n_channels = r.shape[1]

    safe_weights = weights.clone()
    out = torch.zeros((num_voxels, n_channels), device=r.device)
    for k in range(K):
        idx = indices[:, k]
        mask = (idx >= 0) & (idx < num_voxels)
        safe_idx = idx.clone()
        safe_idx[~mask] = 0  # dummy index for invalid
        safe_weights[~mask,k] = 0
        contrib = r * safe_weights[:, k].unsqueeze(1)
        contrib[~mask] = 0
        out.scatter_add_(0, safe_idx.unsqueeze(1).expand(-1, n_channels), contrib)
    return out

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


def compute_diag_LTL(Gx, Gy):
    """
    Efficient diagonal of LᵗL = GxᵗGx + GyᵗGy
    for sparse matrices.
    """
    diag = Gx.pow(2).sum(dim=0).to_dense() + Gy.pow(2).sum(dim=0).to_dense()
    return diag

# def compute_diag_ATA_tensor(stencil, Q_i, num_voxels):
    # """
    # Computes the diagonal of AᵗA for one DWI volume with tensor weighting.
    #
    # Parameters
    # ----------
    # stencil : (indices, weights) for one DWI volume
    # 	indices: [N_particles, K]
    # 	weights: [N_particles, K]
    # Q_i : torch.Tensor [6,]
    # 	Tensor weighting for this DWI (e.g., outer product of gradient direction)
    # num_voxels : int
    # 	Number of voxels (N_voxels)
    #
    # Returns
    # -------
    # diag : torch.Tensor [6 * N_voxels]
    # 	Diagonal preconditioner for AᵗA
    # """
    # indices, weights = stencil
    # N, K = weights.shape
    # device = weights.device
    #
    # diag = torch.zeros(6 * num_voxels, device=device)
    #
    # for k in range(K):
    # 	idx = indices[:, k]
    # 	wk = weights[:, k]
    #
    # 	valid = (idx >= 0) & (idx < num_voxels)
    # 	idx = idx.clone()
    # 	wk = wk.clone()
    # 	idx[~valid] = 0
    # 	wk[~valid] = 0
    #
    # 	# Compute Q_i outer product contributions for each particle
    # 	for d in range(6):
    # 		qd2 = Q_i[d] ** 2  # scalar
    # 		diag_d = wk ** 2 * qd2  # [N]
    # 		diag.scatter_add_(0, idx + d * num_voxels, diag_d)
    #
    # return diag

def compute_diag_ATA_tensor(stencil, Q_i, num_particles):
    indices, weights = stencil         # indices never used!
    device  = weights.device
    w2      = (weights**2).sum(dim=1)  # (N_particles,)

    diag = torch.empty(6 * num_particles, device=device)
    for d in range(6):
        diag[d*num_particles : (d+1)*num_particles] = w2 * (Q_i[d]**2)
    return diag

def compute_diag_ATA_tensor_weighted(stencil, Q_i, img_weight, N_particles):
    """
    img_weight : (N_particles,) – the voxel weight w(p) for this DWI slice
    """
    _, weights = stencil
    w2_pic = (weights**2).sum(dim=1)        # (N_particles,)
    w2_pic *= img_weight                    # ←  multiply by voxel weights

    diag = torch.empty(6*N_particles, device=weights.device)
    for d in range(6):
        diag[d*N_particles : (d+1)*N_particles] = w2_pic * (Q_i[d]**2)
    return diag