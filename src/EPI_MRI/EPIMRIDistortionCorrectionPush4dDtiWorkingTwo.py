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


class EPIMRIDistortionCorrectionPush4dDtiWorkingTwo	:
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
		particle_grids = []
		masks = []

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

			pe_dti_volumes = pe_dti_volumes / pe_b0_mean
			rpe_dti_volumes = rpe_dti_volumes / rpe_b0_mean

			masks.append(pe_mask)
			masks.append(rpe_mask)
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
		self.masks = torch.cat(masks, dim=0)
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

		v = torch.tensor([0.0, 0.0, 1.0], device=self.device,
							 dtype=self.dataObj.dtype)
		bc1 = self.A.mat_mul(yc).reshape(-1, 1)  # averaging matrix & translation vector
		bc1_full = bc1 * v.view(1, -1)  # shift vector in original space

		# Process each PE-RPE pair

		particle_grids = []

		for i, pair in enumerate(self.dataObj.image_pairs):

			xp1 = self.particle_grid.view(3, -1) + bc1_full.T
			xp2 = self.particle_grid.view(3, -1) - bc1_full.T

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

		for slice_index in range(dti_images_lin.shape[1]):

			pic_matrix_vol = []
			pic_matrix_dc = []
			pic_matrix_jac = []
			obs = []

			for image_index in range(dti_images_lin.shape[0]):

				pic_matrix, pic_dc, pic_jac = get_push_forward_matrix_2d_analytic(
					self.dataObj.omega[-4:],
					self.source_m[-2:],
					self.target_m[-2:],
					particle_grids_lin[image_index, :, slice_index].clone(),
					do_derivative=True,
					return_jacobian=True
				)

				pic_matrix = apply_Q_to_pic_matrix(pic_matrix, Q[image_index])

				pic_matrix_vol.append(pic_matrix)
				pic_matrix_dc.append(pic_jac)
				pic_matrix_jac.append(pic_dc)
				obs.append(dti_images_lin[image_index, slice_index].view(-1))

			pic_matrix_vol = torch.cat(pic_matrix_vol, dim=0)
			pic_matrix_dc = torch.stack(pic_matrix_dc, dim=0)

			# obs_image = torch.log(torch.cat(obs, dim=0))
			# invalid_idx = ~torch.isfinite(obs_image)
			# pic_matrix_vol, _ = drop_rows_from_sparse_matrix(pic_matrix_vol, ~invalid_idx)
			# obs_image = obs_image[~invalid_idx]
			#
			# lambda_smooth = 10000000 # or any other value you like
			# sqrt_lambda = torch.sqrt(
			# 	torch.tensor(lambda_smooth, device=pic_matrix_vol.device, dtype=pic_matrix_vol.dtype))
			# Lx, Ly = get_2d_gradient_matrix(self.target_m[-2],
			# 								self.target_m[-1], self.device,
			# 								self.dtype)
			# I6 = torch.eye(6).to_sparse().to(device=self.device,
			# 								 dtype=self.dtype)
			#
			# Gx = sparse_kron(I6, Lx)  # [6*N, 6*N] sparse
			# Gy = sparse_kron(I6, Ly)
			#
			# A_reg = torch.cat([pic_matrix_vol, sqrt_lambda * Gx, sqrt_lambda * Gy], dim=0)
			# b_reg = torch.cat([obs_image.unsqueeze(1),
			# 				   torch.zeros((2 * Gx.shape[0], 1), dtype=obs_image.dtype,
			# 							   device=obs_image.device)], dim=0)
			#
			# rhocorr = tsgu.sparse_lstsq.sparse_generic_lstsq(A_reg, b_reg)






			obs_image = torch.log(torch.cat(obs, dim=0))
			invalid_idx = ~torch.isfinite(obs_image)
			pic_matrix_vol, _ = drop_rows_from_sparse_matrix(pic_matrix_vol,
															 ~invalid_idx)
			obs_image = obs_image[~invalid_idx]

			# Smoothness penalty
			lambda_smooth = 1e7
			sqrt_lambda = torch.sqrt(
				torch.tensor(lambda_smooth, device=pic_matrix_vol.device,
							 dtype=pic_matrix_vol.dtype))

			Lx, Ly = get_2d_gradient_matrix(self.target_m[-2], self.target_m[-1],
											self.device, self.dtype)
			I6 = torch.eye(6).to_sparse().to(device=self.device, dtype=self.dtype)
			Gx = sparse_kron(I6, Lx)
			Gy = sparse_kron(I6, Ly)

			# Define matrix-free operator
			Aop = PicWithRegOperator(pic_matrix_vol, Gx, Gy, sqrt_lambda)

			# Initial guess
			x0 = torch.zeros(Aop.shape[1], device=pic_matrix_vol.device,
							 dtype=pic_matrix_vol.dtype)

			# Right-hand side

			b_data = obs_image.view(-1, 1)  # shape (A_rows, 1)

			# regularization terms → zero vector
			b_reg = torch.zeros((Gx.shape[0] + Gy.shape[0], 1),
								device=b_data.device,
								dtype=b_data.dtype)

			# concatenate in the same order as matvec stacks rows
			b = torch.cat([b_data, b_reg], dim=0)

			# b = obs_image.unsqueeze(1)

			def A_fn(x):
				return Aop.matvec(x)

			# Define RHS: Aᵗ b
			b_rhs = Aop.rmatvec(b)

			# Solve using matrix-free PCG
			solver = PCG(max_iter=50, tol=1e-4, verbose=True)
			x_opt, res, iterOpt, it, resvec = solver.eval(A=A_fn, b=b_rhs,
														  M=None, x=x0)

		# r = rhocorr.reshape(self.dti_images.shape[-2], self.dti_images.shape[-1], 6)
		# 	save_data(r,"/home/laurin/workspace/PyHySCO/data/results/debug/D.nii.gz")
		#
		# 	D_raw = rhocorr.reshape(6, self.dti_images.shape[-2], self.dti_images.shape[-1])
		# 	# r = r.permute(2,0,1)
		# 	save_data(D_raw.permute(2,0,1),"/home/laurin/workspace/PyHySCO/data/results/debug/D2.nii.gz")
		# 	D = D_raw.permute(1,2,0)
		#
		# 	fa = compute_fa_from_tensor(D)
		# 	save_data(fa,
		# 			  "/home/laurin/workspace/PyHySCO/data/results/debug/FA.nii.gz")
		#








			x = 1





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


def apply_Q_to_pic_matrix(A_sparse: torch.Tensor, Q: torch.Tensor):
    """
    Apply tensor weighting Q to a sparse PIC matrix A to produce a sparse design matrix C.

    A_sparse: [N_p, N_vox], torch.sparse_coo_tensor
    Q: [6], tensor encoding (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)

    Returns:
        C_sparse: [N_p, 6 * N_vox], torch.sparse_coo_tensor
    """
    rows, cols = A_sparse.indices()
    values = A_sparse.values()
    N_p, N_vox = A_sparse.shape

    all_rows = []
    all_cols = []
    all_vals = []

    for i in range(6):
        new_cols = cols + i * N_vox       # Offset columns
        new_vals = values * Q[i]          # Scale values

        all_rows.append(rows)
        all_cols.append(new_cols)
        all_vals.append(new_vals)

    rows_cat = torch.cat(all_rows)
    cols_cat = torch.cat(all_cols)
    vals_cat = torch.cat(all_vals)

    C_sparse = torch.sparse_coo_tensor(
        torch.stack([rows_cat, cols_cat]),
        vals_cat,
        size=(N_p, 6 * N_vox)
    ).coalesce()

    return C_sparse


class PicWithRegOperator:
    def __init__(self, A_sparse, Gx, Gy, sqrt_lambda):
        self.A = A_sparse.coalesce()
        self.Gx = Gx.coalesce()
        self.Gy = Gy.coalesce()
        self.sqrt_lambda = sqrt_lambda

        self.num_rows = (
            self.A.shape[0] +
            self.Gx.shape[0] +
            self.Gy.shape[0]
        )
        self.num_cols = self.A.shape[1]  # should be 6*N_vox

    def matvec(self, x):
        # x: (6*N_vox, 1)
        Ax = torch.sparse.mm(self.A, x)
        Gx_term = torch.sparse.mm(self.Gx, x)
        Gy_term = torch.sparse.mm(self.Gy, x)
        return torch.cat([Ax, self.sqrt_lambda * Gx_term, self.sqrt_lambda * Gy_term], dim=0)

    def rmatvec(self, y):
        # y: (total_rows, 1)
        print("y.shape:", y.shape)
        print("Expected:", self.A.shape[0] + self.Gx.shape[0] + self.Gy.shape[0])

        A_rows = self.A.shape[0]
        Gx_rows = self.Gx.shape[0]
        Gy_rows = self.Gy.shape[0]

        y_A = y[:A_rows]
        y_Gx = y[A_rows:A_rows + Gx_rows]
        y_Gy = y[A_rows + Gx_rows:]

        At_y = torch.sparse.mm(self.A.transpose(0, 1), y_A)
        Gxt_y = torch.sparse.mm(self.Gx.transpose(0, 1), y_Gx)
        Gyt_y = torch.sparse.mm(self.Gy.transpose(0, 1), y_Gy)

        return At_y + self.sqrt_lambda * Gxt_y + self.sqrt_lambda * Gyt_y

    @property
    def shape(self):
        return (self.num_rows, self.num_cols)