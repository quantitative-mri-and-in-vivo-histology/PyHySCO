from EPI_MRI.LeastSquaresCorrectionMultiPeSparse4dMasked import \
    LeastSquaresCorrectionMultiPeSparse4dMasked
from optimization.ADMM import *
from EPI_MRI.EPIMRIDistortionCorrection import *
from EPI_MRI.LeastSquaresCorrectionMultiPe import *
from EPI_MRI.LeastSquaresCorrectionMultiPeSparse import *
from EPI_MRI.EPIMRIDistortionCorrectionPush4d import *
from EPI_MRI.EPIMRIDistortionCorrectionPush4dFast import *
from EPI_MRI.EPIMRIDistortionCorrectionPush4dFastBk2 import *
from EPI_MRI.EPIMRIDistortionCorrectionPush4dFastBk3 import *
from EPI_MRI.MultiPeDtiData import MultiPeDtiData
import argparse
import warnings
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="PyHySCO: EPI-MRI Distortion Correction.")
    parser.add_argument("image_config", type=str, help="Path to the input 1 data file (NIfTI format .nii.gz)")
    parser.add_argument("--output_dir", type=str, default=" ", help="Directory to save the corrected images and reports (default=cwd)")
    parser.add_argument("--alpha", type=float, default=300, help="Smoothness regularization parameter (default=300)")
    parser.add_argument("--beta", type=float, default=1e-4, help="Intensity modulation constraint parameter (default=1e-4)")
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness regularization parameter (default=300)")
    parser.add_argument("--rho", type=float, default=1e3, help="Initial Lagrangian parameter (ADMM only) (default=1e3)")
    parser.add_argument("--optimizer", default=GaussNewton, help="Optimizer to use (default=GaussNewton)")
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum number of iterations (default=50)")
    parser.add_argument("--verbose", action="store_true", help="Print details of optimization (default=True)")
    parser.add_argument("--precision", choices=['single', 'double'], default='single', help="Use (single/double) precision (default=single)")
    parser.add_argument("--correction", choices=['jac', 'lstsq'], default='jac', help="Use (Jacobian ['jac']/ Least Squares ['lstsq']) correction (default=lstsq)")
    parser.add_argument("--averaging", default=myAvg1D, help="LinearOperator to use as averaging operator (default=myAvg1D)")
    parser.add_argument("--derivative", default=myDiff1D, help="LinearOperator to use as derivative operator (default=myDiff1D)")
    parser.add_argument("--initialization", default=InitializeCF, help="Initialization method to use (default=InitializeCF)")
    parser.add_argument("--regularizer", default=myLaplacian3D, help="LinearOperator to use for smoothness regularization term (default=myLaplacian3D)")
    parser.add_argument("--PC", default=JacobiCG, help="Preconditioner to use (default=JacobiCG)")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # print(device)

    args = parser.parse_args()
    # defaults
    if args.precision == 'single':
        dtype = torch.float32
    else:
        dtype = torch.float64
    if args.optimizer is ADMM and args.regularizer is not myLaplacian1D:
        warnings.warn("Recommended use is a one-dimensional (in the distortion dimension) regularizer for ADMM.")
    else:
        args.rho = 0.0

    os.makedirs(args.output_dir, exist_ok=True)

    pair_idx = [0,1,2,3]
    # pair_idx = 0
    data = MultiPeDtiData(args.image_config, device=device, dtype=dtype, pair_idx=pair_idx)

    initialization = InitializeCFMultiePeDtiData()
    B0 = initialization.eval(data, blur_result=True)

    B0_detached = B0.detach()

    save_data(B0_detached.reshape(list(m_plus(data.m[1:]))).permute(data.permute_back[0][1:]),
                os.path.join(args.output_dir, 'EstFieldMap.nii.gz'), data.image_pairs[0][0].affine)

    target_res = [128,128]
    loss_func = EPIMRIDistortionCorrectionPush4dFast(
        data,
        alpha=args.alpha,
        beta=args.beta,
        lambda_smooth=args.lambda_smooth,
        target_res=target_res,
        averaging_operator=args.averaging,
        derivative_operator=args.derivative,
        regularizer=args.regularizer,
        rho=args.rho,
        PC=args.PC)

    # change path to be where you want logfile and corrected images to be stored
    # opt = args.optimizer(loss_func, max_iter=args.max_iter, verbose=True, path=args.output_dir)
    opt = args.optimizer(loss_func, max_iter=args.max_iter, verbose=True,
                         path=args.output_dir)
    opt.run_correction(B0)
    opt.apply_correction(method=args.correction)

    # loss_func.eval(B0)

    # avg_operator = myAvg1D(data.omega, data.m, dtype=dtype, device=device)
    # opt = EPIMRIDistortionCorrectionPush4d(data, avg_operator)
    corrected_image = loss_func.recon_image

    scaling = torch.ones(3, dtype=torch.float32, device=device)
    scaling[:2] = data.m[2:] / torch.tensor(target_res, device=data.m.device, dtype=torch.float32)

    target_affine = rescale_affine(data.mats[0], scaling)
    target_affine = target_affine.cpu().numpy()

    save_data(corrected_image.reshape(list(loss_func.m_target)).permute(data.permute_back[0]),
                os.path.join(args.output_dir, 'imCorrected.nii.gz'), target_affine)

    save_data(
        opt.Bc.reshape(list(m_plus(data.m[1:].cpu()))).permute(
            opt.corr_obj.dataObj.permute[0][1:]),
        os.path.join(args.output_dir, 'EstFieldMap.nii.gz'), data.mats[0].cpu().numpy())
    save_data(
        opt.B0.reshape(list(m_plus(data.m[1:].cpu()))).permute(
            opt.corr_obj.dataObj.permute[0][1:]),
        os.path.join(args.output_dir, 'initFieldMap.nii.gz'), data.mats[0].cpu().numpy())


if __name__ == "__main__":
    main()
