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


class EPIMRIDistortionCorrectionPush4dPcg:
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
	def __init__(self, data_obj,
				 alpha,
				 beta,
				 target_res=None,
				 averaging_operator=myAvg1D,
				 derivative_operator=myDiff1D,
				 regularizer=myLaplacian3D,
				 rho=0.0,
				 initialization=InitializeCF,
				 PC=JacobiCG):
		self.dataObj = data_obj
		self.device = data_obj.device
		self.dtype = data_obj.dtype
		self.A = averaging_operator(self.dataObj.omega,self.dataObj.m, self.dtype, self.device)
		self.D = derivative_operator(self.dataObj.omega[-4:], self.dataObj.m[-2:], self.dtype, self.device)
		self.D_vol = derivative_operator(self.dataObj.omega[-2:], self.dataObj.m[-1:], self.dtype, self.device)
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
		alpha = 1e-3
		sse = 0.0

		# --- NEW: Get full coordinate grid (all dims) ---
		xc = get_cell_centered_grid(self.dataObj.omega[2:], self.dataObj.m[1:],
										device=self.device,
									dtype=self.dataObj.dtype,
									return_all=True)

		# Process each PE-RPE pair
		for i, pair in enumerate(self.dataObj.image_pairs):

			dC_pair = []
			Jac_pair = []

			v = torch.tensor([0.0, 0.0, 1.0], device=self.device,
							 dtype=self.dataObj.dtype)
			bc1 = self.A.mat_mul(yc).reshape(-1,
											 1)  # averaging matrix & translation vector
			bc1_full = bc1 * v.view(1, -1)  # shift vector in original space
			xp1 = xc.view(3, -1) + bc1_full.T
			xp2 = xc.view(3, -1) - bc1_full.T

			P = \
				torch.eye(3, device=self.dataObj.device,
						  dtype=self.dataObj.dtype)[
				self.dataObj.permute[i][1:]]
			T = torch.tensor(self.dataObj.rel_mats[i][:3, :3],
						 device=self.dataObj.device,
						 dtype=self.dataObj.dtype)
			T_permuted = P @ T @ P.T

			v_rot = T_permuted @ v
			center = 0.5 * (
					torch.tensor(self.dataObj.omega[3::2]) + torch.tensor(
				self.dataObj.omega[2::2]))  # (x_c, y_c, z_c)

			xp1 = T_permuted @ (xp1 - center.unsqueeze(1)) + center.view(3, 1)
			xp2 = T_permuted @ (xp2 - center.unsqueeze(1)) + center.view(3, 1)

			xp1 = xp1[1:, :].reshape(2, *self.dataObj.m[1:])
			xp2 = xp2[1:, :].reshape(2, *self.dataObj.m[1:])

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

			C1_slices = []
			C2_slices = []
			Jac1_slices = []
			Jac2_slices = []

			for slice_index in range(self.dataObj.m[1]):
				C1_slice, dC1_slice, Jac1_slice = get_push_forward_matrix_2d_analytic(
					self.dataObj.omega[-4:],
					self.dataObj.m[-2:],
					xp1[:, slice_index, :, :].clone(),
					self.dataObj.h[-2:],
					self.dataObj.h[-2:],
					do_derivative=True,
					return_jacobian=True
				)

				C2_slice, dC2_slice, Jac2_slice = get_push_forward_matrix_2d_analytic(
					self.dataObj.omega[-4:],
					self.dataObj.m[-2:],
					xp2[:, slice_index, :, :].clone(),
					self.dataObj.h[-2:],
					self.dataObj.h[-2:],
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

			rho0 = pair.pe_image
			rho1 = pair.rpe_image

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

				# A, row_idx = self.drop_pushforward_matrix_rows(A, thres)
				# b = b[row_idx]

				L = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
				L = L.to_sparse_coo()
				sqrt_lam = torch.sqrt(
					torch.tensor(0.05, dtype=A.dtype, device=A.device))
				A_reg = torch.cat([A, sqrt_lam * L], dim=0)
				b_reg = torch.cat([b,
								   torch.zeros((L.shape[0], 1), dtype=b.dtype,
											   device=b.device)], dim=0)
				b_normal = torch.sparse.mm(A_reg.transpose(0, 1),
										   b_reg).squeeze(1)  # Shape: [16384]

				# rhocorr = tsgu.sparse_lstsq.sparse_generic_lstsq(A_reg, b_reg)


				# rhocorr = sparse_solve_j4t(A_reg, b_reg)
				# rhocorr = sparse_solve_c4t(A_reg, b_reg)

				def A_func(x):
					return torch.sparse.mm(A_reg.transpose(0, 1),
										   torch.sparse.mm(A_reg, x.unsqueeze(
											   1))).squeeze(1)

				with torch.no_grad():
					diag = torch.sparse.mm(
						A_reg.transpose(0, 1),
						torch.sparse.mm(A_reg, torch.ones((A_reg.shape[1], 1), device=A_reg.device))
					).squeeze(1)

				def M_inv(r):
					return r / (
								diag + 1e-6)  # Add epsilon for numerical stability

				pcg = PCG()
				rhocorr = pcg.eval(A_func, b_normal, M_inv)
				rhocorr = rhocorr[0]

				rhocorr_slices.append(rhocorr)
			rhocorr_vol = torch.cat(rhocorr_slices, dim=0)
			rhocorr_vols.append(rhocorr_vol)

		rhocorr = torch.cat(rhocorr_vols, dim=0)
		rhocorr = rhocorr.reshape(*self.dataObj.m)
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

					P = \
						torch.eye(3, device=self.dataObj.device,
								  dtype=self.dataObj.dtype)[
							self.dataObj.permute[i][1:]]
					T = torch.tensor(self.dataObj.rel_mats[i][:3, :3],
									 device=self.dataObj.device,
									 dtype=self.dataObj.dtype)
					T_permuted = P @ T @ P.T

					v_rot = T_permuted @ v

					v_inplane = v_rot[
								1:]  # if you dropped the first (slice) dimension
					# assuming slice dimension is 0 (z), then keep [x, y]

					# Normalize
					v_inplane = v_inplane / torch.norm(v_inplane)
					dy_contrib = dres[..., 0] * v_inplane[0] + dres[..., 1] * \
								 v_inplane[1]

					# dy_contrib = dres[..., 0]

					# dy_contrib = torch.matmul(dres, v_rot)

					dy_sum = dy_contrib.sum(
						dim=1)  # sum contributions from all particles

					dy_sum = dy_sum.to_dense()
					dD_slice = self.D.transp_mat_mul(dy_sum.reshape(self.dataObj.m[2], self.dataObj.m[3]))
					dD_slices.append(dD_slice)

				dD_vol = torch.cat(dD_slices, dim=0)
				dD_vols.append(dD_vol)

			dD_image = torch.cat(dD_vols, dim=0)  # final shape: (m0 * m1 * m2,)
			dD_per_image.append(dD_image.view(self.dataObj.m[0], self.dataObj.m[1], self.dataObj.m[2], -1))

		dD = torch.stack(dD_per_image, dim=0).mean(dim=[0,1])

		Jac_mean = Jac_all.mean(dim=0).reshape(*self.dataObj.m[1:])

		hd = torch.prod(self.dataObj.h)

		# smoothness regularizer
		Sc, dS, d2S = self.S.eval(yc, do_derivative=do_derivative)

		# intensity regularizer
		G, dG, d2G = self.phi_EPI(Jac_mean-1, do_derivative=do_derivative, calc_hessian=calc_hessian)
		Pc = torch.sum(G)

		Dc = 0.5*hd*sse/len(self.dataObj.image_pairs)
		Jc = Dc + hd * self.alpha * Sc + hd * self.beta * Pc
		results = [Jc]

		if do_derivative:
			dP_slices = []
			for slice_index in range(self.dataObj.m[1]):
				dP_slice = self.D.transp_mat_mul(dG[slice_index])
				dP_slices.append(dP_slice)
			dP = torch.stack(dP_slices, dim=0)

			dJ = dD + hd * self.alpha * dS + hd * self.beta * dP
			# save terms of objective function and corrected images
			self.Sc = Sc
			self.Pc = Pc

			results.append(dJ)

		if calc_hessian:
			def H(x):

				d2D_vols = []
				for vol_index in range(self.dataObj.m[0]):

					d2D_slices = []
					for slice_index in range(self.dataObj.m[1]):
						x_slice = x[slice_index]  # shape (m1, m2)
						d2D_slice = torch.zeros_like(x_slice)

						for i in range(len(self.dataObj.image_pairs)):
							dC1_slice, dC2_slice = dC_list[i][slice_index]

							# Apply directional derivative to x_slice
							Dx1 = dC1_slice(x_slice.reshape(-1))  # (H*W, 2)
							Dx2 = dC2_slice(x_slice.reshape(-1))  # (H*W, 2)

							dres = Dx1 - Dx2  # (H*W, 2)
							dy = dres[
								..., -1]  # take PE direction component (e.g., dim 1)
							dy_sum = dy.to_dense().sum(dim=1)  # sum over particles

							# Apply Dᵀ to PE residuals
							d2D_slice += self.D.transp_mat_mul(
								dy_sum.reshape(self.dataObj.m[2], self.dataObj.m[3]))

						d2D_slices.append(d2D_slice)
					d2D_vol	= torch.stack(d2D_slices)
					d2D_vols.append(d2D_vol)

				d2D = torch.stack(d2D_vols)
				d2D = d2D.mean(dim=[0])
				reg_term = self.alpha * self.S.H.mat_mul(x.reshape(-1))
				d2D = d2D + hd * reg_term

				return d2D

			results.append(H)
			M = None
			results.append(M)

		if len(results) == 1:
			return results[0]
		else:
			return results


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



