"""
This file contains various stand-alone functions used throughout the correction process.
"""

import torch
from torch import cuda
import numpy as np
from scipy.io import loadmat
import nibabel as nib
import os
import json


def load_data(im1, im2=None, phase_encoding_direction=1, n=None, device='cpu', dtype=torch.float64):
	"""
	Load original pair of images and return along with relevant parameters.

	Image dimension ordering in returned images:
		- Phase encoding direction is in the last dimension.
		- Frequency encoding direction is in the first dimension (2D) or second dimension (3D)
			or third dimension (4D).
		- Slice selection direction is in the first dimension (3D) or second dimension (4D).
		- Diffusion is in the first dimension (4D).

	Parameters
	----------
	im1 : str
		File path of the first image or stacked image.
	im2 : str, optional
		File path of the second image (default is None, if None assumes stacked image in im1).
	phase_encoding_direction : int, optional
		Specifies which dimension of img1 and img2 is the phase encoding dimension
		(i.e., 1 for the first, 2 for the second, etc.) (default is 1).
	n : int, optional
		Number of diffusion directions to load if 4D input (default is None)
	device : str, optional
		Device on which to compute operations (default is 'cpu').
	dtype : torch.dtype, optional
		Data type for all data tensors (default is torch.float64).


	Returns
	----------
	rho0 : torch.Tensor (shape m)
		First image as a tensor.
	rho1 : torch.Tensor (shape m)
		Second image as a tensor.
	omega : torch.Tensor (size # of dimensions x 2)
		Image domain.
	m : torch.Tensor (size # of dimensions)
		Discretization size.
	h : torch.Tensor (size # of dimensions)
		Cell size.
	permute_back : list (size # of dimensions)
		Order to permute dimensions to return the image to input orientation.
	"""
	file_type_1 = im1.split('.')[-1]

	if file_type_1 == 'mat':
		data = loadmat(im1)
		rho0 = data['I1'].astype(np.float64)
		rho1 = data['I2'].astype(np.float64)
		rho0 = np.rot90(rho0, 1).copy()
		rho1 = np.rot90(rho1, 1).copy()
		rho0 = np.flip(rho0, axis=1).copy()
		rho1 = np.flip(rho1, axis=1).copy()
		permute = [2, 1, 0]
		omega_p = torch.tensor(data['omega'][0], dtype=dtype, device=device)
		omega = torch.zeros_like(omega_p)
		for i in range(len(permute)):
			omega[2 * i:2 * i + 2] = omega_p[2 * permute[i]:2 * permute[i] + 2]
		m = torch.tensor(data['m'][0], dtype=torch.int, device=device)
		if phase_encoding_direction == 1:
			permute = [2, 1, 0]
			permute_back = [2, 1, 0]
		else:
			permute = [2, 0, 1]
			permute_back = [1, 2, 0]
		m = m[permute]
		h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m[permute]
		h = h[permute]

	else:
		file_type_2 = im1.split('.')[-1]
		if file_type_1 != 'gz' or file_type_2 != 'gz':
			raise TypeError("file type(s) not supported")

		if phase_encoding_direction < 1 or phase_encoding_direction > 2:
			raise TypeError("invalid phase encoding direction must be 1 or 2")

		if im2 is None:
			n1_img = nib.load(im1)
			img = n1_img.dataobj
			if img.shape[-1] != 2:
				raise TypeError("expect 3D or 4D stacked image such that last dimension size is 2")
			if len(img.shape) == 3:
				rho0 = np.asarray(img[:, :, 0])
				rho1 = np.asarray(img[:, :, 1])
			elif len(img.shape) == 4:
				rho0 = np.asarray(img[:, :, :, 0])
				rho1 = np.asarray(img[:, :, :, 1])
			elif len(img.shape) == 5:
				rho0 = np.asarray(img[:, :, :, :, 0])
				rho1 = np.asarray(img[:, :, :, :, 1])
			else:
				raise TypeError("expect stacked 2D, 3D, or 4D image")
		else:
			n1_img = nib.load(im1)
			n2_img = nib.load(im2)
			rho0 = np.asarray(n1_img.dataobj)
			rho1 = np.asarray(n2_img.dataobj)

		mat = n1_img.affine

		m = torch.tensor(rho0.shape, dtype=torch.int, device=device)
		dim = torch.numel(m)
		if dim == 2:
			if phase_encoding_direction == 1:
				permute = [1, 0]
				permute_back = [1, 0]
			else:
				permute = [0, 1]
				permute_back = [0, 1]
			# omega = torch.tensor([0, 1, 0, 1], dtype=dtype, device=device)
			# if "2Dexample" in im1:
			# 	omega = torch.tensor([0.0, 239.7126, 0.0, 239.0557], dtype=dtype, device=device)
			# h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
			# m = m[permute]
			# h = h[permute]
			omega_p = torch.zeros(4, dtype=dtype, device=device)
			Vmat = torch.sqrt(torch.sum(torch.tensor(mat, dtype=dtype, device=device)[0:2, 0:2] ** 2, dim=0))
			omega_p[1::2] = Vmat * m
			omega = torch.zeros_like(omega_p)
			for i in range(len(permute)):
				omega[2 * i:2 * i + 2] = omega_p[2 * permute[i]:2 * permute[i] + 2]
			# permute
			m = m[permute]
			h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
		elif dim == 3:
			if phase_encoding_direction == 1:
				permute = [2, 1, 0]
				permute_back = [2, 1, 0]
			else:
				permute = [2, 0, 1]
				permute_back = [1, 2, 0]
			# try:
			# 	omega_p = torch.tensor(loadmat(img_domain)['omega'], dtype=dtype, device=device).flatten()
			# 	omega = torch.zeros_like(omega_p)
			# 	for i in range(len(permute)):
			# 		omega[2 * i:2 * i + 2] = omega_p[2 * permute[i]:2 * permute[i] + 2]
			# 	h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
			# 	h = h[permute]
			# 	m = m[permute]
			# except TypeError:
			# 	omega_p = torch.tensor([0.0000, 209.9840, 0.0000, 185.9524, 0.0000, 122.7389], dtype=dtype, device=device)
			# 	# omega_p = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=dtype, device=device)
			# 	omega = torch.zeros_like(omega_p)
			# 	for i in range(len(permute)):
			# 		omega[2 * i:2 * i + 2] = omega_p[2 * permute[i]:2 * permute[i] + 2]
			# 	m = m[permute]
			# 	h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
			omega_p = torch.zeros(6, dtype=dtype, device=device)
			Vmat = torch.sqrt(torch.sum(torch.tensor(mat, dtype=dtype, device=device)[0:3, 0:3] ** 2, dim=0))
			omega_p[1::2] = Vmat * m
			omega = torch.zeros_like(omega_p)
			for i in range(len(permute)):
				omega[2 * i:2 * i + 2] = omega_p[2 * permute[i]:2 * permute[i] + 2]
			# permute
			m = m[permute]
			h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m

		else:  # dim=4
			if phase_encoding_direction == 1:
				permute = [3, 2, 1, 0]
				permute_back = [3, 2, 1, 0]
			else:
				permute = [3, 2, 0, 1]
				permute_back = [2, 3, 1, 0]
			# omega = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=dtype, device=device)
			# h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
			# m = m[permute]
			# h = h[permute]
			omega = torch.zeros(6, device=device, dtype=dtype)
			Vmat = torch.sqrt(torch.sum(torch.tensor(mat, dtype=dtype, device=device)[0:3, 0:3] ** 2, dim=0))
			# print(Vmat.shape)
			# print(m[1:].shape)
			# print(Vmat)
			# print(m[:-1])
			# print((Vmat * m[:-1]))
			omega[1::2] = (Vmat * m[:-1])[permute[1:]]
			# omega = torch.zeros_like(omega_p)
			# for i in range(len(permute)-1):
			# 	omega[2 * i:2 * i + 2] = omega_p[2 * permute[i]:2 * permute[i] + 2]
			# permute
			m = m[permute]
			if n is not None:
				m[0] = n
				rho0 = rho0[:, :, :, 0:n]
				rho1 = rho1[:, :, :, 0:n]
			omega = torch.hstack((torch.tensor([0, m[0]], dtype=dtype, device=device), omega))
			h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
			# print(omega)
			# print(h)

	rho0 = torch.tensor(rho0, dtype=dtype, device=device)
	rho1 = torch.tensor(rho1, dtype=dtype, device=device)
	rho0 = rho0.permute(permute)
	rho1 = rho1.permute(permute)

	# if 'cuda' in device:
	# 	cuda.empty_cache()

	return rho0, rho1, omega, m, h, permute_back


def get_phase_encoding_direction(json_path):
	"""
	Parse phase encoding direction from JSON file.
	
	Parameters
	----------
	json_path : str
		Path to the JSON file containing phase encoding direction.
		
	Returns
	-------
	int
		Phase encoding direction (1 for first dimension, 2 for second, etc.)
	"""
	try:
		with open(json_path, 'r') as f:
			data = json.load(f)
			
		if 'PhaseEncodingDirection' not in data:
			return 1  # Default to first dimension if not specified
			
		pe_dir = data['PhaseEncodingDirection']
		
		# Map standard direction notation to dimension index
		# i/-i -> 1 (first dimension)
		# j/-j -> 2 (second dimension)
		# k/-k -> 3 (third dimension)
		if pe_dir in ['i', '-i']:
			return 1
		elif pe_dir in ['j', '-j']:
			return 2
		elif pe_dir in ['k', '-k']:
			return 3
		else:
			return 1  # Default to first dimension if unknown
	except (FileNotFoundError, json.JSONDecodeError, KeyError):
		return 1  # Default to first dimension if file not found or invalid
	

class PePair:
	def __init__(self, pe_image, rpe_image):
		self.pe_image = pe_image
		self.rpe_image = rpe_image


def load_data_multi_pe_rpe(images, n=None, device='cpu', dtype=torch.float64):
	"""
	Load original pair of images and return along with relevant parameters.

	Image dimension ordering in returned images:
		- Phase encoding direction is in the last dimension.
		- Frequency encoding direction is in the first dimension (2D) or second dimension (3D)
			or third dimension (4D).
		- Slice selection direction is in the first dimension (3D) or second dimension (4D).
		- Diffusion is in the first dimension (4D).

	Parameters
	----------
	images : list of str
		List of file paths of the input images
	n : int, optional
		Number of diffusion directions to load if 4D input (default is None)
	device : str, optional
		Device on which to compute operations (default is 'cpu').
	dtype : torch.dtype, optional
		Data type for all data tensors (default is torch.float64).

	Returns
	----------
	images : torch.Tensor (shape m)
		Stacked images as a tensor.
	omega : torch.Tensor (size # of dimensions x 2)
		Image domain.
	m : torch.Tensor (size # of dimensions)
		Discretization size.
	h : torch.Tensor (size # of dimensions)
		Cell size.
	permute_back : list (size # of dimensions)
		Order to permute dimensions to return the image to input orientation.
	rel_mats : list of numpy.ndarray
		List of relative transformation matrices for each image.
	"""

	rhos = []
	mats = []
	for image in images:
		img = nib.load(image)
		mat = img.affine
		mats.append(mat)
		rho = np.asarray(img.dataobj)
		rhos.append(rho)

	rel_mats = [np.eye(4)]
	for image_index in range(1, len(images)):
		rel_mats.append(np.linalg.inv(mats[0]) @ mats[image_index])

	# Get shape from first image
	m = torch.tensor(rhos[0].shape, dtype=torch.int, device=device)
	dim = torch.numel(m)

	# Get phase encoding direction from JSON file
	# Use the JSON file with the same base name as the first image
	base_name = os.path.splitext(images[0])[0]
	json_path = base_name + '.json'
	phase_encoding_direction = get_phase_encoding_direction(json_path)

	if dim == 2:
		if phase_encoding_direction == 1:
			permute = [1, 0]
			permute_back = [1, 0]
		else:
			permute = [0, 1]
			permute_back = [0, 1]
		omega_p = torch.zeros(4, dtype=dtype, device=device)
		Vmat = torch.sqrt(torch.sum(torch.tensor(mats[0], dtype=dtype, device=device)[0:2, 0:2] ** 2, dim=0))
		omega_p[1::2] = Vmat * m
		omega = torch.zeros_like(omega_p)
		for i in range(len(permute)):
			omega[2 * i:2 * i + 2] = omega_p[2 * permute[i]:2 * permute[i] + 2]
		m = m[permute]
		h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
	elif dim == 3:
		if phase_encoding_direction == 1:
			permute = [2, 1, 0]
			permute_back = [2, 1, 0]
		else:
			permute = [2, 0, 1]
			permute_back = [1, 2, 0]
		omega_p = torch.zeros(6, dtype=dtype, device=device)
		Vmat = torch.sqrt(torch.sum(torch.tensor(mats[0], dtype=dtype, device=device)[0:3, 0:3] ** 2, dim=0))
		omega_p[1::2] = Vmat * m
		omega = torch.zeros_like(omega_p)
		for i in range(len(permute)):
			omega[2 * i:2 * i + 2] = omega_p[2 * permute[i]:2 * permute[i] + 2]
		m = m[permute]
		h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
	else:  # dim=4
		if phase_encoding_direction == 1:
			permute = [3, 2, 1, 0]
			permute_back = [3, 2, 1, 0]
		else:
			permute = [3, 2, 0, 1]
			permute_back = [2, 3, 1, 0]
		omega = torch.zeros(6, device=device, dtype=dtype)
		Vmat = torch.sqrt(torch.sum(torch.tensor(mats[0], dtype=dtype, device=device)[0:3, 0:3] ** 2, dim=0))
		omega[1::2] = (Vmat * m[:-1])[permute[1:]]
		m = m[permute]
		if n is not None:
			m[0] = n
			for i in range(len(rhos)):
				rhos[i] = rhos[i][:, :, :, 0:n]
		omega = torch.hstack((torch.tensor([0, m[0]], dtype=dtype, device=device), omega))
		h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m

	# Stack images and apply permutation
	rhos = torch.stack([torch.tensor(rho, dtype=dtype, device=device) for rho in rhos], dim=0)
	rhos = rhos.permute([0] + [i+1 for i in permute])  # Add 1 to permute indices since we added a new dimension

	return rhos, omega, m, h, permute_back, rel_mats


def save_data(data, filepath, affine=None):
	"""
	Save data to the given filepath.

	Parameters
	----------
	data : torch.Tensor (size any)
		Data to save.
	filepath : str
		Path where to save data.
	"""
	if affine is None:
		affine = np.eye(4)
	if filepath is not None:
		save_img = nib.Nifti1Image(np.asarray(data.cpu()), affine)
		nib.save(save_img, filepath)


def normalize(im1, im2):
	"""
	Normalize the pair of image intensities to [0, 256].

	Parameters
	----------
	im1 : torch.Tensor (size any)
		First image.
	im2 : torch.Tensor (size any)
		Second image.

	Returns
	----------
	i1 : torch.Tensor (size same as im1)
		Normalized first image.
	i2 : torch.Tensor (size same as im2)
		Normalized second image.
	"""
	min_i = torch.min(torch.min(im1), torch.min(im2))
	i1 = im1 - min_i
	i2 = im2 - min_i
	max_i = torch.max(torch.max(i1), torch.max(i2))
	i1 = (256 / max_i) * i1
	i2 = (256 / max_i) * i2
	return i1, i2

def normalize_multi_pe(images):
	"""
	Normalize a stack of images to have the same mean intensity.

	Parameters
	----------
	images : torch.Tensor
		Stack of images to normalize, with first dimension being the image index.

	Returns
	-------
	torch.Tensor
		Normalized stack of images.
	"""
	# Calculate mean intensity for each image
	means = torch.mean(images, dim=tuple(range(1, images.dim())))
	
	# Calculate the mean of all image means
	mean_mean = torch.mean(means)
	
	# Normalize each image by scaling its mean to match the overall mean
	scale_factors = mean_mean / means
	scale_factors = scale_factors.view(-1, *([1] * (images.dim() - 1)))
	
	return images * scale_factors


def m_plus(m):
	"""
	Given dimensions m of the original image, return augmented dimensions (plus one in phase encoding direction).

	Parameters
	----------
	m : torch.Tensor (size # of dimensions)
		Original dimensions.

	Returns
	----------
	m2 : torch.Tensor (size # of dimensions)
		Augmented dimensions.
	"""
	m2 = m.clone()
	m2[-1] = m2[-1] + 1
	return m2


def m_minus(m):
	"""
	Given dimensions m of the augmented image, return original dimensions.

	Parameters
	----------
	m : torch.Tensor (size # of dimensions)
		Augmented dimensions.

	Returns
	----------
	m2 : torch.Tensor (size # of dimensions)
		Original dimensions.
	"""
	m2 = m.clone()
	m2[-1] = m2[-1] - 1
	return m2


def interp_parallel(x, y, xs, device='cpu'):
	"""
	Vectorized interpolation - parallelized in the first (or first and second if 3D) dimension.

	Parameters
	----------
	x : torch.Tensor (size (dim, num_points))
		Given points.
	y : torch.Tensor (size (1, num_points))
		Given function values.
	xs : torch.Tensor (size (dim, num_new_points))
		Points at which to interpolate.
	device : str, optional
		Device on which to compute operations (default is 'cpu').

	Returns
	----------
	interpolation : torch.Tensor (size (1, num_new_points))
		Function values at xs.
	"""
	i = torch.searchsorted(x[:, 1:].contiguous(), xs.contiguous())
	j = torch.arange(x.shape[0]).view(x.shape[0], 1).expand_as(i)
	x_hat = torch.empty(xs.shape, device=device)
	x_hat = torch.div(torch.sub(xs, x[j, i], out=x_hat), torch.sub(x[j, i + 1], x[j, i]), out=x_hat)
	ret = torch.empty(x_hat.shape, device=device)
	ret = torch.add(torch.mul(y[j, i], torch.sub(1, x_hat, out=ret), out=ret), torch.mul(y[j, i + 1], x_hat), out=ret)
	return ret.squeeze(dim=0)


def get_cell_centered_grid(omega, m, device='cpu', dtype=torch.float64, return_all=False):
	"""
	Generate the cell-centered grid of size m over domain omega.

	Parameters
	----------
	omega : torch.Tensor (size # of dimensions x 2)
		Image domain.
	m : torch.Tensor (size # of dimensions)
		Discretization size.
	device : str, optional
		Device on which to compute operations (default is 'cpu').
	dtype : torch.dtype, optional
		Data type for all data tensors (default is torch.float64).
	return_all : bool, optional
		Flag to return grid in non-distortion dimensions (default is False).

	Returns
	----------
	x : torch.Tensor (size (prod(m),1) or size (# of dimensions * prod(m),1) if return_all=True)
		Cell-centered grid in the distortion dimension or all dimensions if return_all=True.
	"""
	h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m

	def xi(i):
		t = torch.linspace(omega[2 * i - 2] + h[i - 1] / 2.0, omega[2 * i - 1] - h[i - 1] / 2.0, int(m[i - 1]), device=device, dtype=dtype)
		t = torch.reshape(t, (1, -1))
		return torch.transpose(t, 0, 1).squeeze()

	if torch.numel(m) == 1:
		return xi(0)

	x = None
	if not return_all:
		if torch.numel(m) == 2:
			x = torch.meshgrid(xi(1), xi(2), indexing='ij')[1]
			x = x.reshape(-1, 1)
		elif torch.numel(m) == 3:
			x = torch.meshgrid(xi(1), xi(2), xi(3), indexing='ij')[2]
			x = x.reshape(-1, 1)
		elif torch.numel(m) == 4:
			x = torch.meshgrid(xi(1), xi(2), xi(3), xi(4), indexing='ij')[3]
			x = x.reshape(-1, 1)

	else:
		if torch.numel(m) == 2:
			x1, x2 = torch.meshgrid(xi(1), xi(2), indexing='ij')
			x = torch.cat((torch.reshape(x1, (-1, 1)), torch.reshape(x2, (-1, 1))))
		elif torch.numel(m) == 3:
			x1, x2, x3 = torch.meshgrid(xi(1), xi(2), xi(3), indexing='ij')
			x = torch.cat((torch.reshape(x1, (-1, 1)), torch.reshape(x2, (-1, 1)), torch.reshape(x3, (-1, 1))))
		elif torch.numel(m) == 4:
			x1, x2, x3, x4 = torch.meshgrid(xi(1), xi(2), xi(3), xi(4), indexing='ij')
			x = torch.cat((torch.reshape(x1, (-1, 1)), torch.reshape(x2, (-1, 1)), torch.reshape(x3, (-1, 1)), torch.reshape(x4, (-1, 1))))
	return x


def get_pe_rpe_images(image_path_1, image_path_2):
	"""
	Determine which image is PE and which is RPE based on their phase encoding directions.
	
	Parameters
	----------
	image_path_1 : str
		Path to first image
	image_path_2 : str
		Path to second image
		
	Returns
	-------
	image_path_pe : str
		Path to the PE image (positive phase encoding direction)
	image_path_rpe : str
		Path to the RPE image (negative phase encoding direction)
	phase_encoding_direction : int
		Phase encoding direction (1, 2, or 3)
	"""
	# Get phase encoding directions from JSON files
	# Handle .nii.gz extension by removing both .nii and .gz
	base_name_1 = os.path.splitext(os.path.splitext(image_path_1)[0])[0]
	base_name_2 = os.path.splitext(os.path.splitext(image_path_2)[0])[0]
	json_path_1 = base_name_1 + '.json'
	json_path_2 = base_name_2 + '.json'
	
	# Get phase encoding direction from JSON files
	with open(json_path_1, 'r') as f:
		json_data_1 = json.load(f)
	with open(json_path_2, 'r') as f:
		json_data_2 = json.load(f)
	
	pe_dir_1 = json_data_1['PhaseEncodingDirection']
	pe_dir_2 = json_data_2['PhaseEncodingDirection']
	
	# Get the axis and sign from the phase encoding direction string
	pe_axis_1 = pe_dir_1[-1]  # Last character is the axis (i, j, k)
	pe_axis_2 = pe_dir_2[-1]
	pe_sign_1 = -1 if pe_dir_1.startswith('-') else 1
	pe_sign_2 = -1 if pe_dir_2.startswith('-') else 1
	
	# Convert axis to number (i=1, j=2, k=3)
	axis_map = {'i': 1, 'j': 2, 'k': 3}
	pe_axis_1 = axis_map[pe_axis_1]
	pe_axis_2 = axis_map[pe_axis_2]
	
	# Determine which is PE and which is RPE
	if pe_axis_1 == pe_axis_2:  # Same axis
		if pe_sign_1 == 1 and pe_sign_2 == -1:
			image_path_pe = image_path_1
			image_path_rpe = image_path_2
			phase_encoding_direction = pe_axis_1
		elif pe_sign_1 == -1 and pe_sign_2 == 1:
			image_path_pe = image_path_2
			image_path_rpe = image_path_1
			phase_encoding_direction = pe_axis_1
		else:
			raise ValueError(f"Invalid phase encoding signs: {pe_dir_1}, {pe_dir_2}. Expected one to be positive and the other negative.")
	else:
		raise ValueError(f"Different phase encoding directions: {pe_dir_1}, {pe_dir_2}. Expected same direction.")
	
	return image_path_pe, image_path_rpe, phase_encoding_direction


def sparse_kron_mat_int(Q, N_p):
    """
    Computes the sparse Kronecker product of Q and identity matrix of size N_p.

    Q: Tensor of shape (N_dir, 6)
    Returns sparse matrix of shape (N_dir*N_p, 6*N_p)
    """
    device = Q.device
    dtype = Q.dtype

    N_dir, D = Q.shape
    total_rows = N_dir * N_p
    total_cols = D * N_p

    # Build row and col indices
    q_row = torch.arange(N_dir, device=device).repeat_interleave(D)
    q_col = torch.arange(D, device=device).repeat(N_dir)
    q_val = Q.flatten()

    # Repeat each row block for each voxel
    row_idx = (q_row[:, None] * N_p + torch.arange(N_p, device=device)).reshape(-1)
    col_idx = (q_col[:, None] * N_p + torch.arange(N_p, device=device)).reshape(-1)
    val = q_val.repeat_interleave(N_p)

    indices = torch.stack([row_idx, col_idx], dim=0)
    B = torch.sparse_coo_tensor(indices, val, size=(total_rows, total_cols), device=device, dtype=dtype)
    return B.coalesce()


# Let:
# A: (N_vox, N_par) (sparse)
# Q: (N_dir, 6)
# B: (N_dir*N_vox, 6*N_par) from sparse_kron_mat_int(Q, N_vox)

def sparse_kron_eye_mat(A, N_dir):
    """
    Builds a sparse Kronecker product of I (size N_dir) with sparse matrix A.
    Result is block diagonal with A repeated N_dir times.
    Output shape: (N_dir * A.shape[0], N_dir * A.shape[1])
    """
    A = A.coalesce()
    row_A, col_A = A.indices()
    val_A = A.values()

    n_rows, n_cols = A.shape
    total_rows = N_dir * n_rows
    total_cols = N_dir * n_cols

    row_idx = []
    col_idx = []
    val = []

    for i in range(N_dir):
        row_shift = i * n_rows
        col_shift = i * n_cols
        row_idx.append(row_A + row_shift)
        col_idx.append(col_A + col_shift)
        val.append(val_A)

    row_idx = torch.cat(row_idx)
    col_idx = torch.cat(col_idx)
    val = torch.cat(val)

    indices = torch.stack([row_idx, col_idx], dim=0)
    return torch.sparse_coo_tensor(indices, val, size=(total_rows, total_cols), device=A.device, dtype=A.dtype).coalesce()

def permute_affine_axes(affine, perm):
    """
    Permute the axes of an affine matrix according to the given permutation.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine matrix (e.g. from a NIfTI image)
    perm : list or array of length 3
        Permutation of axes, e.g., [0, 2, 1] means:
        - original x becomes x
        - original y becomes z
        - original z becomes y

    Returns
    -------
    permuted_affine : np.ndarray
        The 4x4 affine matrix with permuted axes.
    """
    permuted_affine = torch.eye(4, dtype=affine.dtype, device=affine.device)

    # Rotation + scaling part
    R = affine[:3, :3]
    permuted_affine[:3, :3] = R[perm, :][:, perm]

    # Translation part
    t = affine[:3, 3]
    permuted_affine[:3, 3] = t[perm]

    return permuted_affine

def rescale_affine(A_src, scale):
    """
    Rescale affine to reflect resampling from m_source to m_target in 3D.

    Parameters
    ----------
    A_src : (4, 4) np.ndarray
        Original affine matrix of the source image.
    m_source : tuple of int
        Shape (D, H, W) or (Z, Y, X) of the source image.
    m_target : tuple of int
        Shape of the target image.

    Returns
    -------
    A_tgt : (4, 4) np.ndarray
        New affine matrix for the target image.
    """
    A_tgt = A_src.clone()
    A_tgt[:3, :3] = A_src[:3, :3] @ torch.diag(scale)
    return A_tgt


def compute_fa_from_tensor(D):
    """
    Compute fractional anisotropy (FA) from a tensor field stored as [H, W, 6].

    Parameters
    ----------
    D : torch.Tensor
        Tensor of shape [H, W, 6] containing diffusion tensor components:
        [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] per voxel.

    Returns
    -------
    FA : torch.Tensor
        Fractional anisotropy map of shape [H, W].
    """
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = [D[..., i] for i in range(6)]

    # Reconstruct full tensor [H, W, 3, 3]
    tensor = torch.stack([
        torch.stack([Dxx, Dxy, Dxz], dim=-1),
        torch.stack([Dxy, Dyy, Dyz], dim=-1),
        torch.stack([Dxz, Dyz, Dzz], dim=-1)
    ], dim=-2)  # shape [H, W, 3, 3]

    eigvals = torch.linalg.eigvalsh(tensor)  # shape [H, W, 3]
    md = eigvals.mean(dim=-1)

    sqrt_3_over_2 = torch.sqrt(torch.tensor(1.5, device=D.device, dtype=D.dtype))
    numerator = sqrt_3_over_2 * torch.linalg.norm(eigvals - md.unsqueeze(-1), dim=-1)
    denominator = torch.linalg.norm(eigvals, dim=-1)

    fa = numerator / (denominator + 1e-12)  # Avoid division by zero

    return fa

def sparse_kron(A, B):
    """
    Compute the sparse Kronecker product of two sparse matrices A ⊗ B.

    Parameters
    ----------
    A : torch.sparse_coo_tensor of shape [m, n]
    B : torch.sparse_coo_tensor of shape [p, q]

    Returns
    -------
    torch.sparse_coo_tensor of shape [m*p, n*q]
    """
    A = A.coalesce()
    B = B.coalesce()

    rows_A, cols_A = A.indices()
    rows_B, cols_B = B.indices()

    vals_A = A.values()
    vals_B = B.values()

    # Kronecker indices
    row_idx = (rows_A[:, None] * B.shape[0] + rows_B[None, :]).reshape(-1)
    col_idx = (cols_A[:, None] * B.shape[1] + cols_B[None, :]).reshape(-1)

    # Kronecker values (outer product of A and B values)
    val_idx = (vals_A[:, None] * vals_B[None, :]).reshape(-1)

    indices = torch.stack([row_idx, col_idx], dim=0)
    shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    return torch.sparse_coo_tensor(indices, val_idx, shape, device=A.device, dtype=A.dtype)
