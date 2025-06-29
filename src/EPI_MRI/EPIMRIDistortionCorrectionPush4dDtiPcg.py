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
# from torchsparsegradutils.sparse_lstsq import cg, lsmr


class EPIMRIDistortionCorrectionPush4dDtiPcg:
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
				 target_res = None,
				 averaging_operator=myAvg1D,
				 derivative_operator=myDiff1D,
				 regularizer=myLaplacian3D,
				 rho=0.0,
				 initialization=InitializeCF,
				 PC=JacobiCG):
		self.dataObj = data_obj
		self.device = data_obj.device
		self.dtype = data_obj.dtype

		self.source_m = self.dataObj.m
		self.source_omega = self.dataObj.omega
		self.source_h = self.dataObj.h

		if target_res is None:
			self.target_m = self.dataObj.m
		else:
			self.target_m = torch.tensor([*self.source_m[:-2], *target_res], dtype=self.source_m.dtype, device=self.device)
			self.target_omega = self.source_omega
			self.target_h = self.source_h/self.target_m*self.source_m


		self.A = averaging_operator(self.dataObj.omega[2:], self.dataObj.m[1:], self.dtype, self.device)
		self.D_slice = derivative_operator(self.dataObj.omega[-4:], self.dataObj.m[-2:], self.dtype, self.device)
		self.D_image = derivative_operator(self.dataObj.omega[-4:], self.target_m[-2:], self.dtype, self.device)
		self.D = derivative_operator(self.dataObj.omega[-6:], self.dataObj.m[-3:], self.dtype, self.device)
		self.xc = get_cell_centered_grid(self.dataObj.omega, self.dataObj.m, device=self.device, dtype=self.dtype).reshape(tuple(self.dataObj.m))
		self.S = QuadRegularizer(regularizer(self.dataObj.omega[2:],self.dataObj.m[1:], self.dtype, self.device))
		self.Q = TikRegularizer(self.dataObj.omega, self.dataObj.m)
		self.alpha = alpha
		self.beta = beta
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

		dti_images = []
		pe_dirs = []
		bvec = []
		bval = []
		rel_mats = []
		masks = []
		particle_grids = []

		for i, pair in enumerate(self.dataObj.image_pairs):

			pe_image = pair[0]
			rpe_image = pair[1]
			pe_mask = pair[0].mask
			rpe_mask = pair[1].mask


			pe_bvec = torch.tensor(pe_image.bvec, dtype=self.dtype, device=self.device)
			rpe_bvec = torch.tensor(rpe_image.bvec, dtype=self.dtype, device=self.device)
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
			pe_b0_mean = torch.mean(pe_b0_volumes, dim = 0)
			rpe_b0_mean = torch.mean(rpe_b0_volumes, dim = 0)


			pe_dti_volumes = pe_image.data[pe_dti_idx]
			rpe_dti_volumes = rpe_image.data[rpe_dti_idx]
			pe_mask_repeated = pe_mask.unsqueeze(0).repeat(pe_dti_volumes.shape[0], 1, 1, 1)
			rpe_mask_repeated = pe_mask.unsqueeze(0).repeat(rpe_dti_volumes.shape[0], 1, 1, 1)

			masks.append(pe_mask_repeated)
			masks.append(rpe_mask_repeated)

			pe_dti_volumes = pe_dti_volumes / pe_b0_mean
			rpe_dti_volumes = rpe_dti_volumes / rpe_b0_mean

			dti_images.append(pe_dti_volumes)
			dti_images.append(rpe_dti_volumes)
			bvec.append(pe_bvec[:,pe_dti_idx].T)
			bvec.append(rpe_bvec[:,rpe_dti_idx].T)
			bval.append(pe_bval[pe_dti_idx])
			bval.append(rpe_bval[rpe_dti_idx])
			pe_dirs.append(torch.tensor(pe_image.phase_sign, dtype=self.dtype, device=self.device))
			pe_dirs.append(torch.tensor(rpe_image.phase_sign, dtype=self.dtype, device=self.device))
			rel_mats.append(self.dataObj.rel_mats[i])
			rel_mats.append(self.dataObj.rel_mats[i])

		self.dti_images = torch.stack(dti_images, dim=0)
		self.masks = torch.stack(masks, dim=0)
		self.bvec = torch.stack(bvec, dim=0)
		self.bval = torch.stack(bval, dim=0)
		self.rel_mats = torch.stack(rel_mats, dim=0)
		self.pe_dirs = torch.tensor(pe_dirs, dtype=self.dtype, device=self.device)
		self.particle_grid = get_cell_centered_grid(self.dataObj.omega[2:],
									self.dataObj.m[1:],
									device=self.device,
									dtype=self.dataObj.dtype,
									return_all=True)



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
		C_list = []
		rho_list = []
		Jac_list = []
		dC_list = []
		sse = 0.0

		lambda_smooth = 5000000  # or any other value you like
		# sqrt_lambda = torch.sqrt(
		#     torch.tensor(lambda_smooth, device=A.device, dtype=A.dtype))
		Lx, Ly = get_2d_gradient_matrix(self.target_m[-2],
										self.target_m[-1], self.device,
										self.dtype)

		I6 = torch.eye(6).to_sparse().to(device=self.device,
										 dtype=self.dtype)

		Gx = sparse_kron(I6, Lx)  # [6 * N_x_edges, 6 * N_voxels]
		Gy = sparse_kron(I6, Ly)  # [6 * N_y_edges, 6 * N_voxels]

		diag_reg = compute_diag_LTL(Gx, Gy)


		v = torch.tensor([0.0, 0.0, 1.0], device=self.device,
							 dtype=self.dataObj.dtype)
		bc1 = self.A.mat_mul(yc).reshape(-1, 1)  # averaging matrix & translation vector
		bc1_full = bc1 * v.view(1, -1)  # shift vector in original space

		# Process each PE-RPE pair

		particle_grids = []


		for i, pair in enumerate(self.dataObj.image_pairs):

			xp1 = self.particle_grid.view(3, -1) + bc1_full.T
			xp2 = self.particle_grid.view(3, -1) - bc1_full.T

			# xp1 = self.particle_grid.view(3, -1)
			# xp2 = self.particle_grid.view(3, -1)

			T_permuted = self.dataObj.rel_mats[i][:3, :3]

			center = 0.5 * (
					torch.tensor(self.dataObj.omega[3::2]) + torch.tensor(
				self.dataObj.omega[2::2]))  # (x_c, y_c, z_c)

			xp1 = T_permuted @ (xp1 - center.unsqueeze(1)) + center.view(3, 1)
			xp2 = T_permuted @ (xp2 - center.unsqueeze(1)) + center.view(3, 1)

			xp1 = xp1[1:, :].reshape(2, *self.dataObj.m[1:])
			xp2 = xp2[1:, :].reshape(2, *self.dataObj.m[1:])

			particle_grids.append(xp1.repeat(self.dti_images.shape[1], 1, 1, 1, 1))
			particle_grids.append(xp2.repeat(self.dti_images.shape[1], 1, 1, 1, 1))

		particle_grids = torch.stack(particle_grids, dim=0)

		particle_grids_lin = particle_grids.view(-1, *particle_grids.shape[2:])
		dti_images_lin = self.dti_images.view(-1, *self.dti_images.shape[2:])
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

			pic_matrix_vol = []
			pic_matrix_dc = []
			pic_matrix_jac = []
			obs = []
			obs_masks = []
			stencil_weights_vol = []
			stencil_indices_vol = []

			for image_index in range(dti_images_lin.shape[0]):

				# pic_matrix, pic_dc, pic_jac = get_push_forward_matrix_2d_analytic(
				#     self.dataObj.omega[-4:],
				#     self.source_m[-2:],
				#     self.target_m[-2:],
				#     particle_grids_lin[image_index, :, slice_index].clone(),
				#     do_derivative=True,
				#     return_jacobian=True
				# )

				stencils = build_pic_stencils_2d(self.dataObj.omega[-4:],
												  self.source_m[-2:],
												  self.target_m[-2:],
												  particle_grids_lin[image_index, :, slice_index].clone())
				stencil_indices_vol.append(stencils[0])
				stencil_weights_vol.append(stencils[1])

				# pic_matrix = apply_Q_to_pic_matrix(pic_matrix, Q[image_index])

				# pic_matrix_vol.append(pic_matrix)
				# pic_matrix_dc.append(pic_jac)
				# pic_matrix_jac.append(pic_dc)
				obs.append(dti_images_lin[image_index, slice_index].view(-1))
				obs_masks.append(masks_lin[image_index, slice_index].view(-1))
				# obs.append(dti_images_lin[image_index, slice_index])

			# pic_matrix_vol = torch.cat(pic_matrix_vol, dim=0)
			# pic_matrix_dc = torch.stack(pic_matrix_dc, dim=0)

			stencil_indices_vol = torch.stack(stencil_indices_vol, dim=0)
			stencil_weights_vol = torch.stack(stencil_weights_vol, dim=0)


			obs_image = torch.stack(obs, dim=0)
			save_data(obs_image.reshape(-1, 66, 66),"/home/laurin/workspace/PyHySCO/data/results/debug/obs.nii.gz")
			obs_image = torch.log(obs_image)
			save_data(obs_image.reshape(-1, 66, 66),
					  "/home/laurin/workspace/PyHySCO/data/results/debug/obs_log.nii.gz")

			obs_masks = torch.stack(obs_masks, dim=0)


			# t = 1
			#
			# # obs_img_view_list = []
			# # for k in range(obs_image.shape[0]):
			# # 	obs_img_view_list.append(obs_image[k, :].view(66,66))
			# # obs_img_view = torch.stack(obs_img_view_list, dim=0)
			#


			# obs_image = torch.log(obs_image)
			# b = rho_all[vol_index, slice_index].unsqueeze(1)
			# all_b = obs_image
			# # all_b = all_b.permute(1, 0)
			#
			# for b_im_index, b_im in enumerate(all_b):
			# 	save_data(b_im.reshape(66, 66).permute(0, 1),
			# 			  f"/home/laurin/workspace/PyHySCO/data/results/debug/b_{b_im_index}.nii.gz")

			# all_b = b.view(-1, 2*len(self.dataObj.image_pairs))
			# all_b = all_b.permute(1, 0)
			# def A_fn(x):
			#     with torch.no_grad():
			#         result = torch.zeros_like(x)
			#         for stencils in stencils_per_slice:
			#             Ax = apply_pic_matvec(stencils, x)  # A_k x
			#             AtAx = apply_pic_rmatvec(stencils, Ax,
			#                                      x.shape[0])  # A_k^T A_k x
			#             result += AtAx
			#         return result

			# obs_image[obs_image > 1.0] = torch.nan
			# obs_image[~obs_masks] = torch.nan
			# obs_image[~obs_image.isfinite()] = 1e-6

			valid_mask = (obs_image < 1.0) & obs_masks & torch.isfinite(obs_image)
			obs_image[~valid_mask] = 0.0

			roi_stencils = []
			for k in range(stencil_indices_vol.shape[0]):
				stencil = (stencil_indices_vol[k, :, :],
						   stencil_weights_vol[k, :, :])
				roi_stencils.append(stencil)

			def A_fn(x):
				result = torch.zeros_like(x)

				for i in range(obs_image.shape[0]):
					Q_i = Q[i]  # shape (6,)
					stencil = roi_stencils[i]  # particle-to-voxel stencil for this DWI


					indices, weights = stencil
					weights = weights.clone()
					weights[~valid_mask[i],:] = 0

					stencil_filtered = (indices, weights)
					Ax_i = apply_pic_tensor_matvec(stencil_filtered, x,
												   Q_i)  # A_i x with Q weighting
					AtAx_i = apply_pic_tensor_rmatvec(stencil_filtered, Ax_i, Q_i, self.target_m[-2]*self.target_m[-1])
					result += AtAx_i

				# Add smoothness regularization
				if lambda_smooth > 0:
					Dx = Gx @ x  # gradient in x-dir
					Dy = Gy @ x  # gradient in y-dir
					LTLx = Gx.T @ Dx + Gy.T @ Dy
					result += lambda_smooth * LTLx

				return result

			with torch.no_grad():
				rhs = torch.zeros((6 * self.target_m[-2]*self.target_m[-1], 1),
								  device=self.device)

				for i in range(obs_image.shape[0]):
					Q_i = Q[i]
					stencil = roi_stencils[i]
					b_i = obs_image[i].clone().unsqueeze(1)  # [N, 1]

					indices, weights = stencil
					weights = weights.clone()
					weights[~valid_mask[i],:] = 0
					stencil_filtered = (indices, weights)

					b_i[~valid_mask[i]] = 0.0

					# compute Aᵀb
					Atb_i = apply_pic_tensor_rmatvec(
						stencil_filtered, b_i, Q_i, self.target_m[-2]*self.target_m[-1]
					)
					rhs += Atb_i

			x0 = torch.zeros_like(rhs)

			# def make_diag_preconditioner(diag):
			# 	diag = diag.view(-1, 1)  # ensure it's column vector
			#
			# 	def M(x):
			# 		x = x.view(-1, 1)  # ensure x is also column vector
			# 		return x / (diag + 1e-6)  # safe: [N, 1] / [N, 1]
			#
			# 	return M
			#

			def make_diag_preconditioner(diag):
				diag = diag.view(-1, 1)

				def M(x):
					return x / diag

				return M



			diag_data = torch.zeros((6 * self.target_m[-2]*self.target_m[-1]), device=self.device)

			for i in range(dti_images_lin.shape[0]):
				Q_i = Q[i]  # shape (6,)
				stencil = roi_stencils[i]
				diag_data += compute_diag_ATA_tensor(stencil, Q_i, self.target_m[-2]*self.target_m[-1])

			precond_diag = diag_data + lambda_smooth * diag_reg

			M = make_diag_preconditioner(precond_diag)


			#
			# diag = torch.zeros(self.target_m[-2] * self.target_m[-1],
			# 				   device=self.device)
			# for stencils in roi_stencils:
			# 	diag += compute_diag_ATA(stencils,
			# 							 self.target_m[-2] * self.target_m[
			# 								 -1])

			# diag = compute_diag_ATA(stencils, self.target_m[-2]*self.target_m[-1])
			# M = make_diag_preconditioner(diag)

			solver = PCG(max_iter=200, tol=1e-2, verbose=True)
			rhocorr, _, _, _, _ = solver.eval(A=A_fn, b=rhs, M=M, x=x0)

			# save_data(rhocorr.reshape(66, 66),
			#           "/home/laurin/workspace/PyHySCO/data/results/debug/rhocorr_dti_dwi.nii.gz")

			D_raw = rhocorr.reshape(6, self.target_m[-2], self.target_m[-1])
			# r = r.permute(2,0,1)
			# save_data(D_raw.permute(2,0,1),"/home/laurin/workspace/PyHySCO/data/results/debug/D2.nii.gz")
			D = D_raw.permute(1,2,0)

			fa = compute_fa_from_tensor(D)
			# save_data(fa,
			#           "/home/laurin/workspace/PyHySCO/data/results/debug/FA.nii.gz")

			rhocorr_slices.append(D)
			fa_slices.append(fa)


		rhocorr_image = torch.stack(rhocorr_slices, dim=0)

		save_data(rhocorr_image.permute(*self.dataObj.permute_back[0][1:], 3),"/home/laurin/workspace/PyHySCO/data/results/debug/D.nii.gz")
		fa_image = torch.stack(fa_slices, dim=0)
		save_data(fa_image.permute(*self.dataObj.permute_back[0][1:]),
				  "/home/laurin/workspace/PyHySCO/data/results/debug/FA.nii.gz")


		# Process each PE-RPE pair
		for i, pair in enumerate(self.dataObj.image_pairs):

			dC_pair = []

			v = torch.tensor([0.0, 0.0, 1.0], device=self.device,
							 dtype=self.dataObj.dtype)
			bc1 = self.A.mat_mul(yc).reshape(-1,
											 1)  # averaging matrix & translation vector
			bc1_full = bc1 * v.view(1, -1)  # shift vector in original space
			xp1 = xc.view(3, -1) + bc1_full.T
			xp2 = xc.view(3, -1) - bc1_full.T

			T_permuted = self.dataObj.rel_mats[i][:3, :3]

			center = 0.5 * (
					torch.tensor(self.dataObj.omega[3::2]) + torch.tensor(
				self.dataObj.omega[2::2]))  # (x_c, y_c, z_c)

			xp1 = T_permuted @ (xp1 - center.unsqueeze(1)) + center.view(3, 1)
			xp2 = T_permuted @ (xp2 - center.unsqueeze(1)) + center.view(3, 1)

			xp1 = xp1[1:, :].reshape(2, *self.dataObj.m[1:])
			xp2 = xp2[1:, :].reshape(2, *self.dataObj.m[1:])


			C1_slices = []
			C2_slices = []
			Jac1_slices = []
			Jac2_slices = []

			for slice_index in range(self.dataObj.m[1]):
				C1_slice, dC1_slice, Jac1_slice = get_push_forward_matrix_2d_analytic(
					self.dataObj.omega[-4:],
					self.source_m[-2:],
					self.target_m[-2:],
					xp1[:, slice_index, :, :].clone(),
					do_derivative=True,
					return_jacobian=True
				)

				C2_slice, dC2_slice, Jac2_slice = get_push_forward_matrix_2d_analytic(
					self.dataObj.omega[-4:],
					self.source_m[-2:],
					self.target_m[-2:],
					xp2[:, slice_index, :, :].clone(),
					do_derivative=True,
					return_jacobian=True
				)

				C1_slices.append(C1_slice)
				C2_slices.append(C2_slice)
				dC_pair.append((dC1_slice, dC2_slice))
				Jac1_slices.append(Jac1_slice)
				Jac2_slices.append(Jac2_slice)

			dC_list.append(dC_pair)
			Jac1_vol = torch.stack(Jac1_slices, dim=0)
			Jac2_vol = torch.stack(Jac2_slices, dim=0)

			Jac_list.append(Jac1_vol)
			Jac_list.append(Jac2_vol)

			C1 = torch.stack(C1_slices, dim=0)
			C2 = torch.stack(C2_slices, dim=0)

			C = torch.cat((C1, C2), dim=1)

			# Store matrices and data
			C_list.append(C)

			rho0 = pair[0].data
			rho1 = pair[1].data

			rho0 = rho0.reshape(self.dataObj.m[0], self.dataObj.m[1], -1)
			rho1 = rho1.reshape(self.dataObj.m[0], self.dataObj.m[1], -1)

			rho_list.append(torch.cat((rho0, rho1), dim=-1))

		# Concatenate all matrices and data
		C_all = torch.cat(C_list, dim=1)
		rho_all = torch.cat(rho_list, dim=-1)
		Jac_all = torch.stack(Jac_list, dim=0)

		rhocorr_vols = []
		for vol_index in range(self.dataObj.m[0]):
			rhocorr_slices = []
			for slice_index in range(self.dataObj.m[1]):
				A = C_all[slice_index]
				b = rho_all[vol_index, slice_index].unsqueeze(1)

				lambda_smooth = 0.2  # or any other value you like
				sqrt_lambda = torch.sqrt(
					torch.tensor(lambda_smooth, device=A.device, dtype=A.dtype))
				Lx, Ly = get_2d_gradient_matrix(self.target_m[-2], self.target_m[-1], self.device, self.dtype)
				A_reg = torch.cat([A, sqrt_lambda*Lx, sqrt_lambda*Ly], dim=0)
				b_reg = torch.cat([b,
								   torch.zeros((2*Lx.shape[0], 1), dtype=b.dtype,
											   device=b.device)], dim=0)

				rhocorr = tsgu.sparse_lstsq.sparse_generic_lstsq(A_reg, b_reg)
				rhocorr_slices.append(rhocorr)
			rhocorr_vol = torch.cat(rhocorr_slices, dim=0)
			rhocorr_vols.append(rhocorr_vol)

		rhocorr = torch.cat(rhocorr_vols, dim=0)
		rhocorr = rhocorr.reshape(*self.target_m)
		self.recon_image = rhocorr


		# Precompute for reuse
		hd = torch.prod(self.dataObj.h)


		# Loop over each image pair and slice
		dD_per_image = []
		for i in range(len(self.dataObj.image_pairs)):

			dD_vols = []
			for vol_index in range(self.dataObj.m[0]):
				dD_slices = []
				for slice_index in range(self.dataObj.m[1]):
					# Get the relevant dC and Jacobians for this pair/slice
					dC1_slice, dC2_slice = dC_list[i][slice_index]

					# Get PE/RPE slice from corrected image (computed from LS)
					rhoc = rhocorr[vol_index,slice_index].reshape(-1)

					# Compute residuals: predicted - observed
					res1 = dC1_slice(rhoc)  # shape (H*W, 2)
					res2 = dC2_slice(rhoc)  # same shape

					# Combine contributions
					# This assumes 2D dC returns torch.stack([dx, dy], dim=-1)
					# You must apply chain rule back to 1D field in phase encoding direction
					dres = res1 - res2  # shape: (H*W, N, 2
					pe_dim = -1  # assume this is 1
					dy_contrib = dres[..., pe_dim]  # shape: (H*W, N)

					v = torch.tensor([0.0, 0.0, 1.0], device=self.device,
									 dtype=self.dataObj.dtype)
					T_permuted = self.dataObj.rel_mats[i][:3, :3]

					v_rot = T_permuted @ v

					v_inplane = v_rot[1:]

					# Normalize
					v_inplane = v_inplane / torch.norm(v_inplane)
					dy_contrib = dres[..., 0] * v_inplane[0] + dres[..., 1] * \
								 v_inplane[1]

					# dy_contrib = dres[..., 0]

					# dy_contrib = torch.matmul(dres, v_rot)

					dy_sum = dy_contrib.sum(
						dim=1)  # sum contributions from all particles

					dy_sum = dy_sum.to_dense()
					dD_slice = self.D_slice.transp_mat_mul(dy_sum.reshape(self.dataObj.m[2], self.dataObj.m[3]))
					dD_slices.append(dD_slice)

				dD_vol = torch.cat(dD_slices, dim=0)
				dD_vols.append(dD_vol)

			dD_image = torch.cat(dD_vols, dim=0)  # final shape: (m0 * m1 * m2,)
			dD_per_image.append(dD_image.view(self.dataObj.m[0], self.dataObj.m[1], self.dataObj.m[2], -1))

		dD = torch.stack(dD_per_image, dim=0).mean(dim=[0,1])
		geom = self.A.mat_mul(dD)

		Jac_mean = Jac_all.mean(dim=0).reshape(*self.source_m[1:])

		intensity = Jac_mean


		# compute distance measure
		hd_source = torch.prod(self.source_h)
		hd_target = torch.prod(self.target_h)

		Dc = sse / (len(self.dataObj.image_pairs) * 2)

		# smoothness regularizer
		Sc, dS, d2S = self.S.eval(yc, do_derivative=do_derivative)

		# intensity regularizer
		dbc = self.D.mat_mul(
			yc)  # derivative matrix & derivative of deformation vector
		Jac = 1 + dbc  # determinant of the transform xc+bc
		G, dG, d2G = self.phi_EPI(Jac - 1, do_derivative=do_derivative,
								  calc_hessian=calc_hessian)
		Pc = torch.sum(G)
		dP = None
		if do_derivative:
			dP = self.D.transp_mat_mul(dG)

		# compute proximal term
		if self.rho > 0:
			Qc, dQ, d2Q = self.Q.eval(yc, 1.0, yref, do_derivative=do_derivative)
		else:
			Qc = 0.0
			dQ = 0.0
			d2Q = None

		# save terms of objective function and corrected images
		self.Dc = Dc
		self.Sc = Sc
		self.Pc = Pc
		self.Qc = Qc

		Jc = hd_target * Dc + hd_source * self.alpha * Sc + hd_source * self.beta * Pc + self.rho * Qc
		if not do_derivative:
			return Jc
		dJ = hd_target * dD + hd_source * self.alpha * dS + hd_source * self.beta * dP + self.rho * dQ
		if not calc_hessian:
			return Jc, dJ
		else:
			def H(x):
				""" Matrix-vector product between Hessian and a tensor x of size m_plus(m). """
				Dx = self.D.mat_mul(x)
				dr = geom * Dx + intensity * self.A.mat_mul(x)
				dr_d2psi = dr * hd
				if self.beta == 0:  # d2P is zeros
					d2D = self.D.transp_mat_mul(
						dr_d2psi * geom) + self.A.transp_mat_mul(
						dr_d2psi * intensity)
					return d2D + hd * self.alpha * d2S.mat_mul(
						x) + hd * self.rho * x
				else:
					d2D = self.D.transp_mat_mul(
						dr_d2psi * geom + hd * self.beta * d2G * Dx) + self.A.transp_mat_mul(
						dr_d2psi * intensity)
					return d2D + hd * self.alpha * d2S.mat_mul(
						x) + hd * self.rho * x

			if self.PC is not None:
				diagD, diagP, diagS = self.PC.getM(geom, intensity, hd, d2G, self.D,
												   self.A, self.S.H, self.alpha,
												   self.beta)
				self.PC.M += hd * self.rho
				M = lambda x: self.PC.eval(x)
			else:
				M = None

			return Jc, dJ, H, M



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
		x2 = x*x
		G = torch.nan_to_num(-(x2*x2) / (x2-1))
		if do_derivative:
			dG = torch.nan_to_num(-2*(x*x2)*(x2-2) / (x2-1)**2)
		if calc_hessian:
			d2G = torch.nan_to_num(-2 * x2 * (x2 * x2 - 3 * x2 + 6) / (x2 - 1) ** 3)
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

	N = H * W
	idx = torch.arange(N, device=device)

	# Gradient in x-direction (horizontal)
	mask_x = (idx + 1) % W != 0  # exclude last column
	rows_x = torch.arange(mask_x.sum(), device=device)
	cols_x = idx[mask_x]
	cols_x_right = cols_x + 1
	data_x = torch.stack([-torch.ones_like(rows_x), torch.ones_like(rows_x)])
	indices_x = torch.stack([rows_x.repeat(2), torch.cat([cols_x, cols_x_right])])
	Lx = torch.sparse_coo_tensor(indices_x, data_x.flatten(), size=(mask_x.sum(), N), device=device, dtype=dtype)

	# Gradient in y-direction (vertical)
	mask_y = idx < (H - 1) * W  # exclude last row
	rows_y = torch.arange(mask_y.sum(), device=device)
	cols_y = idx[mask_y]
	cols_y_down = cols_y + W
	data_y = torch.stack([-torch.ones_like(rows_y), torch.ones_like(rows_y)])
	indices_y = torch.stack([rows_y.repeat(2), torch.cat([cols_y, cols_y_down])])
	Ly = torch.sparse_coo_tensor(indices_y, data_y.flatten(), size=(mask_y.sum(), N), device=device, dtype=dtype)

	return Lx.coalesce(), Ly.coalesce()


def build_pic_stencils_2d(omega, m_source, m_target, xp):
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
	num_voxels : int
		Total number of voxels in the image.
	"""
	device = xp.device

	# source grid settings
	n_particles = torch.prod(m_source)
	particle_size = (omega[1::2] - omega[0::2]) / m_source

	# target grid settings
	n_cells = torch.prod(m_target)
	cell_size = (omega[1::2] - omega[0::2]) / m_target

	x_vox = (xp[0] - omega[0]) / cell_size[0]
	y_vox = (xp[1] - omega[2]) / cell_size[1]

	x_vox = x_vox.permute(1, 0)
	y_vox = y_vox.permute(1, 0)


	Px = torch.ceil(x_vox).long()
	wx = x_vox - (Px.float()-0.5)
	Py = torch.ceil(y_vox).long()
	wy = y_vox - (Py.float()-0.5)

	# Px = torch.floor(x_vox).long()
	# wx = x_vox - Px.float()
	# Py = torch.floor(y_vox).long()
	# wy = y_vox - Py.float()
	Px = Px.reshape(-1)
	Py = Py.reshape(-1)
	wx = wx.reshape(-1)
	wy = wy.reshape(-1)

	pwidth = torch.ceil(particle_size / cell_size).to(torch.int32)
	pwidth = [5,5]

	# # # # Evaluate 1D basis
	Bx, Dx = int1DSingle(wx, pwidth[0], particle_size[0], cell_size[0],
						 particle_size[0], do_derivative=True)  # [2*p+1, N]
	By, Dy = int1DSingle(wy, pwidth[1], particle_size[1], cell_size[1],
						 particle_size[1], do_derivative=True)

	# Bx, Dx = int1DSingle(wx, pwidth[0], cell_size[0], particle_size[0], cell_size[0], do_derivative=do_derivative)  # [2*p+1, N]
	# By, Dy = int1DSingle(wy, pwidth[1], cell_size[1], particle_size[1], cell_size[1], do_derivative=do_derivative)

	Bx = Bx / Bx.sum(0)
	By = By / By.sum(0)

	nbx = Bx.shape[0]
	nby = By.shape[0]
	nVoxel = nbx * nby

	I = torch.empty(nVoxel * n_particles, dtype=torch.long, device=device)
	J = torch.empty(nVoxel * n_particles, dtype=torch.long, device=device)
	B = torch.empty(nVoxel * n_particles, dtype=xp.dtype, device=device)

	pp = 0
	for i, px in enumerate(range(-pwidth[0], pwidth[0] + 1)):
		for j, py in enumerate(range(-pwidth[1], pwidth[1] + 1)):


			idx = slice(pp * n_particles, (pp + 1) * n_particles)
			pp += 1

			x_idx = Px + px
			y_idx = Py + py
			Iij = x_idx * m_target[0] + y_idx  # Flattened linear index
			#Iij = y_idx * m_target[1] + x_idx

			Bij = Bx[i, :] * By[j, :]  # Elementwise per-particle weight

			I[idx] = Iij
			J[idx] = torch.arange(n_particles, device=Px.device)
			B[idx] = Bij


	stencil_size = nbx * nby

	indices = I.view(stencil_size, n_particles)
	weights = B.view(stencil_size, n_particles)
	indices = indices.permute(1, 0)
	weights = weights.permute(1, 0)
	stencils = (indices, weights)

	return stencils


def apply_pic_tensor_matvec(stencils, x, Q):
	indices, weights = stencils  # [N, K]
	N, K = weights.shape
	num_vox = x.shape[0] // 6

	x = x.view(6, num_vox)  # [6, N_vox]

	# Output: r = sum_j w_ij * sum_k Q_k * x_k[indices_ij]
	r = torch.zeros((N,), device=x.device)

	for k in range(6):
		xk = x[k]  # shape [N_vox]
		r += Q[k] * apply_pic_matvec(stencils, xk.unsqueeze(1)).squeeze()

	return r.unsqueeze(1)  # shape [N, 1]



def apply_pic_tensor_rmatvec(stencils, r, Q, num_voxels):
	indices, weights = stencils
	N, K = weights.shape
	n_channels = r.shape[1]

	assert n_channels == 1, "This version assumes r is (N, 1)"
	out = torch.zeros((6 * num_voxels, 1), device=r.device)

	for k in range(K):
		idx = indices[:, k]
		mask = (idx >= 0) & (idx < num_voxels)
		safe_idx = idx.clone()
		safe_idx[~mask] = 0
		w = weights[:, k].clone()
		w[~mask] = 0

		for i in range(6):  # 6 tensor components
			scaled = r * (w * Q[i]).unsqueeze(1)
			out_idx = safe_idx + i * num_voxels
			out.scatter_add_(0, out_idx.unsqueeze(1), scaled)
			t = 1

	return out


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


def compute_diag_ATA_tensor(stencil, Q_i, num_voxels):
	"""
	Computes the diagonal of AᵗA for one DWI volume with tensor weighting.

	Parameters
	----------
	stencil : (indices, weights) for one DWI volume
		indices: [N_particles, K]
		weights: [N_particles, K]
	Q_i : torch.Tensor [6,]
		Tensor weighting for this DWI (e.g., outer product of gradient direction)
	num_voxels : int
		Number of voxels (N_voxels)

	Returns
	-------
	diag : torch.Tensor [6 * N_voxels]
		Diagonal preconditioner for AᵗA
	"""
	indices, weights = stencil
	N, K = weights.shape
	device = weights.device

	diag = torch.zeros(6 * num_voxels, device=device)

	for k in range(K):
		idx = indices[:, k]
		wk = weights[:, k]

		valid = (idx >= 0) & (idx < num_voxels)
		idx = idx.clone()
		wk = wk.clone()
		idx[~valid] = 0
		wk[~valid] = 0

		# Compute Q_i outer product contributions for each particle
		for d in range(6):
			qd2 = Q_i[d] ** 2  # scalar
			diag_d = wk ** 2 * qd2  # [N]
			diag.scatter_add_(0, idx + d * num_voxels, diag_d)

	return diag


def compute_diag_LTL(Gx, Gy):
	"""
	Efficient diagonal of LᵗL = GxᵗGx + GyᵗGy
	for sparse matrices.
	"""
	diag = Gx.pow(2).sum(dim=0).to_dense() + Gy.pow(2).sum(dim=0).to_dense()
	return diag
