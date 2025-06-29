import torch
import json
import os
import nibabel as nib
import numpy as np
from EPI_MRI.utils import normalize, permute_affine_axes
import pandas as pd



class MultiPeDtiData:

    def __init__(self, image_config, device='cpu', dtype=torch.float64,
                 pair_idx=None, do_normalize=True):
        self.image_config = image_config
        self.image_pairs = []
        self.omega = []
        self.m = []
        self.h = []
        self.permute_back = []
        self.permute = []
        self.rel_mats = []
        self.device = device
        self.dtype = dtype
        self.pair_idx = pair_idx
        self.image_pairs, self.omega, self.m, self.h, self.permute, self.permute_back, self.mats, self.rel_mats = self.load_data()

        if do_normalize:
            for i in range(len(self.image_pairs)):
                self.image_pairs[i][0].data, self.image_pairs[
                    i][1].data = normalize(self.image_pairs[i][0].data,
                                             self.image_pairs[i][1].data)

    def load_data(self):
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
        device = self.device
        dtype = self.dtype

        with open(self.image_config, 'r') as f:
            image_config = json.load(f)

        image_dir = os.path.dirname(self.image_config)

        pe_rpe_image_paths = []
        pe_rpe_image_pairs = []
        pe_rpe_phase_encoding_directions = []
        pe_rpe_mats = []
        pe_rpe_rel_mats = []
        pe_rpe_permute_back = []
        pe_rpe_permute = []
        n_slice_idx = list(range(11, 14))
        n_slice_idx = None
        #n_vol_idx = list(range(0, 8))
        n_vol_idx = None

        # image_config['pe_pairs'] = [image_config['pe_pairs'][0]]
        # image_config['pe_pairs'] = image_config['pe_pairs'][0:2]

        if self.pair_idx is None:
            pass
        elif isinstance(self.pair_idx, list):
            image_pairs = [image_config['pe_pairs'][self.pair_idx] for
                           self.pair_idx in self.pair_idx]
            image_config['pe_pairs'] = image_pairs
        else:
            image_config['pe_pairs'] = [image_config['pe_pairs'][self.pair_idx]]

        for image_pair_index, image_path_pair in enumerate(
                image_config['pe_pairs']):

            image_path_1 = os.path.join(image_dir, image_path_pair[0])
            image_path_2 = os.path.join(image_dir, image_path_pair[1])

            dwi_series_pe = load_dwi_pe_series(image_path_1)
            dwi_series_rpe = load_dwi_pe_series(image_path_2)

            phase_encoding_direction = dwi_series_pe.phase_axis

            # Store the phase encoding direction
            pe_rpe_phase_encoding_directions.append(phase_encoding_direction)

            rho0 = dwi_series_pe.data
            rho1 = dwi_series_rpe.data

            mask0 = dwi_series_pe.mask
            mask1 = dwi_series_rpe.mask

            mat = dwi_series_pe.affine

            mat = torch.tensor(mat, dtype=dtype, device=device)
            pe_rpe_mats.append(mat)
            rel_mat = torch.linalg.inv(pe_rpe_mats[image_pair_index]) @ pe_rpe_mats[0]
            # rel_mat = pe_rpe_mats[0] @ torch.linalg.inv(pe_rpe_mats[image_pair_index])

            # rel_mat = np.linalg.inv(pe_rpe_mats[0] @ pe_rpe_mats[image_pair_index])
            rel_mat = torch.tensor(rel_mat, dtype=dtype, device=device)

            m = torch.tensor(rho0.shape, dtype=torch.int, device=device)
            dim = torch.numel(m)
            if dim == 2:
                if phase_encoding_direction == 1:
                    permute = [1, 0]
                    permute_back = [1, 0]
                else:
                    permute = [0, 1]
                    permute_back = [0, 1]
                omega_p = torch.zeros(4, dtype=dtype, device=device)
                Vmat = torch.sqrt(torch.sum(
                    torch.tensor(mat, dtype=dtype, device=device)[0:2,
                    0:2] ** 2, dim=0))
                omega_p[1::2] = Vmat * m
                omega = torch.zeros_like(omega_p)
                for i in range(len(permute)):
                    omega[2 * i:2 * i + 2] = omega_p[
                                             2 * permute[i]:2 * permute[i] + 2]
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
                omega_p = torch.zeros(6, dtype=dtype, device=device)
                Vmat = torch.sqrt(torch.sum(
                    torch.tensor(mat, dtype=dtype, device=device)[0:3,
                    0:3] ** 2, dim=0))
                omega_p[1::2] = Vmat * m
                omega = torch.zeros_like(omega_p)
                for i in range(len(permute)):
                    omega[2 * i:2 * i + 2] = omega_p[
                                             2 * permute[i]:2 * permute[i] + 2]
                # permute
                m = m[permute]
                if n_slice_idx is not None:
                    m[0] = len(n_slice_idx)
                    rho0 = rho0[:, :, n_slice_idx]
                    rho1 = rho1[:, :, n_slice_idx]
                    if mask0 is not None:
                        mask0 = mask0[:, :, n_slice_idx]
                    if mask1 is not None:
                        mask1 = mask1[:, :, n_slice_idx]

                h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m

            else:  # dim=4
                if phase_encoding_direction == 1:
                    permute = [3, 2, 1, 0]
                    permute_back = [3, 2, 1, 0]
                else:
                    permute = [3, 2, 0, 1]
                    permute_back = [2, 3, 1, 0]
                omega = torch.zeros(6, device=device, dtype=dtype)
                Vmat = torch.sqrt(torch.sum(
                    torch.tensor(mat, dtype=dtype, device=device)[0:3,
                    0:3] ** 2, dim=0))
                omega[1::2] = (Vmat * m[:-1])[permute[1:]]
                m = m[permute]
                if n_slice_idx is not None:
                    m[1] = len(n_slice_idx)
                    rho0 = rho0[:, :, n_slice_idx, :]
                    rho1 = rho1[:, :, n_slice_idx, :]
                    if mask0 is not None:
                        mask0 = mask0[:, :, n_slice_idx]
                    if mask1 is not None:
                        mask1 = mask1[:, :, n_slice_idx]
                if n_vol_idx is not None:
                    m[0] = len(n_vol_idx)
                    dwi_series_pe.bvec = dwi_series_pe.bvec[:,n_vol_idx]
                    dwi_series_rpe.bvec = dwi_series_pe.bvec[:,n_vol_idx]
                    dwi_series_pe.bval = dwi_series_pe.bval[n_vol_idx]
                    dwi_series_rpe.bval = dwi_series_pe.bval[n_vol_idx]
                    rho0 = rho0[:, :, :, n_vol_idx]
                    rho1 = rho1[:, :, :, n_vol_idx]


                omega = torch.hstack((torch.tensor([0, m[0]], dtype=dtype,
                                                   device=device), omega))
                h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m

            rho0 = torch.tensor(rho0, dtype=dtype, device=device)
            rho1 = torch.tensor(rho1, dtype=dtype, device=device)
            rho0 = rho0.permute(permute)
            rho1 = rho1.permute(permute)

            if mask0 is not None:
                mask0 = torch.tensor(mask0, dtype=torch.bool, device=device)
                mask0 = mask0.permute(permute[1:])
            else:
                mask0 = torch.ones(rho0.shape[1:], dtype=torch.bool,
                                   device=device)
            dwi_series_pe.mask = mask0

            if mask1 is not None:
                mask1 = torch.tensor(mask1, dtype=torch.bool, device=device)
                mask1 = mask1.permute(permute[1:])
            else:
                mask1 = torch.ones(rho1.shape[1:], dtype=torch.bool,
                                   device=device)
            dwi_series_rpe.mask = mask1

            dwi_series_pe.data = rho0
            dwi_series_rpe.data = rho1
            dwi_series_pe.data = rho0
            dwi_series_rpe.data = rho1

            pe_rpe_image_pairs.append((dwi_series_pe, dwi_series_rpe))
            pe_rpe_permute_back.append(permute_back)
            pe_rpe_permute.append(permute)

            rel_mat = permute_affine_axes(rel_mat, permute[1:])

            pe_rpe_rel_mats.append(rel_mat)


        return pe_rpe_image_pairs, omega, m, h, pe_rpe_permute, pe_rpe_permute_back, pe_rpe_mats, pe_rpe_rel_mats




class DwiPeSeries:
    def __init__(self, data, affine, phase_axis, phase_sign, bval, bvec, mask=None):
        """
        Container for a single DWI acquisition and its metadata.

        Parameters
        ----------
        data : np.ndarray
            The image data (from NIfTI), shape (X, Y, Z, N)
        affine : np.ndarray
            The 4x4 affine matrix from the NIfTI image
        phase_axis : int
            Phase encoding axis (1=i, 2=j, 3=k)
        phase_sign : int
            Phase encoding direction sign (+1 or -1)
        bvals : np.ndarray
            Array of b-values, shape (N,)
        bvecs : np.ndarray
            Array of b-vectors, shape (3, N)
        """
        self.data = data
        self.mask = mask
        self.affine = affine
        self.phase_axis = phase_axis
        self.phase_sign = phase_sign
        self.bval = bval
        self.bvec = bvec

    def __repr__(self):
        return (f"DwiPeSeries(phase_axis={self.phase_axis}, "
                f"phase_sign={self.phase_sign}, "
                f"data.shape={self.data.shape}, "
                f"bvals.shape={self.bvals.shape}, "
                f"bvecs.shape={self.bvecs.shape})")


def load_dwi_pe_series(image_path):
    """
    Load a DWI image and associated metadata into a DwiPeSeries object.

    Parameters
    ----------
    image_path : str
        Path to the DWI image (.nii or .nii.gz)

    Returns
    -------
    DwiPeSeries
        Object containing image data, affine, and associated PE/bval/bvec metadata.
    """
    base_path = os.path.splitext(os.path.splitext(image_path)[0])[0]
    json_path = base_path + ".json"
    bval_path = base_path + ".bval"
    bvec_path = base_path + ".bvec"
    mask_path = base_path + "_mask.nii"

    # Load NIfTI image
    img = nib.load(image_path)
    data = img.get_fdata()
    affine = img.affine

    if os.path.exists(mask_path):
        mask = nib.load(mask_path)
        mask = mask.get_fdata().astype(np.bool)
    else:
        mask = None

    # Load JSON metadata
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    pe_dir = json_data["PhaseEncodingDirection"]
    axis_map = {"i": 1, "j": 2, "k": 3}
    phase_sign = -1 if pe_dir.startswith('-') else 1
    phase_axis = axis_map[pe_dir[-1]]

    # Load bvals and bvecs
    bval = np.loadtxt(bval_path)
    bvec = np.loadtxt(bvec_path)

    # bval_df = pd.read_csv(bval_path, header=None)
    # bvec_df = pd.read_csv(bvec_path)
    #
    # bval = bval_df.to_numpy().squeeze()
    # bvec = bvec_df.to_numpy()
    #
    # if bval.ndim > 1:
    #     if bval.shape[0] == 1:
    #         bval = bval[0]
    #
    # if bvec.shape[0] != 3:
    #     bvec = bvec.T

    return DwiPeSeries(
        data=data,
        affine=affine,
        phase_axis=phase_axis,
        phase_sign=phase_sign,
        bval=bval,
        bvec=bvec,
        mask=mask
    )
