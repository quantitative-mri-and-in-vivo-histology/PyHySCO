import argparse
import warnings

import torch

from EPI_MRI.EPIMRIDistortionCorrection import *
from EPI_MRI.EPIMRIDistortionCorrectionLsq import EPIMRIDistortionCorrectionLsq
from EPI_MRI.EPIMRIDistortionCorrectionLsq4d import EPIMRIDistortionCorrectionLsq4d
from EPI_MRI.InitializationMethods import *
from optimization.ADMM import *
from optimization.LBFGS import *
from EPI_MRI.utils import *
from EPI_MRI.ParticleInCell2D import *



def main():

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    dtype = torch.float32


    omega = torch.tensor([0, 1, 0, 1], dtype=torch.int, device=device)
    m_small = torch.tensor([10, 10], dtype=torch.int, device=device)
    m_big = torch.tensor([20, 20], dtype=torch.int, device=device)
    # m_small = m_big

    xc = get_cell_centered_grid(omega, m_small, device=device, dtype=dtype, return_all=True)
    # xc = xc.reshape(2,10,10)
    # xc = xc.permute(0,2,1)
    xc = xc.reshape(2,-1)

    B = torch.tensor([0, 1, 4, 0, 1, 0])
    I = torch.tensor([1, 1, 1, 1, 1, 1])
    J = torch.tensor([0, 0, 0, 0, 0, 0])

    T = torch.sparse_coo_tensor(
        torch.stack((I, J)), B, size=(3,3)).coalesce()

    mat, dmat = get_push_forward_matrix_2d_analytic(omega, m_small, m_big, xc, device=device, do_derivative=True)
    mat = mat.to_dense()

    t = 1

    save_data(mat, os.path.join("/home/laurin/workspace/PyHySCO/data/results/4d_highres_ds_gn/T.nii.gz"))


if __name__ == "__main__":
    main()
