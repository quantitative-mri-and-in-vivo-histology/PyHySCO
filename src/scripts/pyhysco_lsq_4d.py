from EPI_MRI.LeastSquaresCorrectionMultiPeSparse4dMasked import \
    LeastSquaresCorrectionMultiPeSparse4dMasked
from optimization.ADMM import *
from EPI_MRI.EPIMRIDistortionCorrection import *
from EPI_MRI.LeastSquaresCorrectionMultiPe import *
from EPI_MRI.LeastSquaresCorrectionMultiPeSparse import *
from EPI_MRI.LeastSquaresCorrectionMultiPeSparse4d import *
import argparse
import warnings
import torch.nn.functional as F


class MultiPeDataObject:
    def __init__(self, image_config, device='cpu', dtype=torch.float64, pair_idx=None, first_vol_only=False):
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
        self.first_vol_only = first_vol_only
        self.image_pairs, self.omega, self.m, self.h, self.permute, self.permute_back, self.rel_mats = self.load_data()

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

        n = None

        # image_config['pe_pairs'] = [image_config['pe_pairs'][0]]
        # image_config['pe_pairs'] = image_config['pe_pairs'][0:2]

        if self.pair_idx is None:
            pass
        elif isinstance(self.pair_idx, list):
            image_pairs = [image_config['pe_pairs'][self.pair_idx] for self.pair_idx in self.pair_idx]
            image_config['pe_pairs'] = image_pairs
        else:
            image_config['pe_pairs'] = [image_config['pe_pairs'][self.pair_idx]]
        
        for image_pair_index, image_path_pair in enumerate(image_config['pe_pairs']):

            image_path_1 = os.path.join(image_dir, image_path_pair[0])
            image_path_2 = os.path.join(image_dir, image_path_pair[1])

            # Get PE and RPE images and phase encoding direction
            image_path_pe, image_path_rpe, phase_encoding_direction = get_pe_rpe_images(image_path_1, image_path_2)
            
            # Store the phase encoding direction
            pe_rpe_phase_encoding_directions.append(phase_encoding_direction)
            
            # Store the PE and RPE image paths
            pe_rpe_image_paths.append([image_path_pe, image_path_rpe])

            n1_img = nib.load(image_path_1)
            n2_img = nib.load(image_path_2)
            rho0 = np.asarray(n1_img.dataobj)
            rho1 = np.asarray(n2_img.dataobj)

            if self.first_vol_only and rho1.ndim == 4:
                rho0 = rho0[:, :, :, 1]
                rho1 = rho1[:, :, :, 1]

            mat = n1_img.affine
            pe_rpe_mats.append(mat)
            #pe_rpe_rel_mats.append(np.linalg.inv(pe_rpe_mats[0]) @ pe_rpe_mats[image_pair_index])
            pe_rpe_rel_mats.append(np.linalg.inv(pe_rpe_mats[image_pair_index]) @ pe_rpe_mats[0])

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
                omega = torch.zeros(6, device=device, dtype=dtype)
                Vmat = torch.sqrt(torch.sum(torch.tensor(mat, dtype=dtype, device=device)[0:3, 0:3] ** 2, dim=0))
                omega[1::2] = (Vmat * m[:-1])[permute[1:]]
                m = m[permute]
                if n is not None:
                    m[0] = n
                    rho0 = rho0[:, :, :, 0:n]
                    rho1 = rho1[:, :, :, 0:n]
                omega = torch.hstack((torch.tensor([0, m[0]], dtype=dtype, device=device), omega))
                h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m

            rho0 = torch.tensor(rho0, dtype=dtype, device=device)
            rho1 = torch.tensor(rho1, dtype=dtype, device=device)
            rho0 = rho0.permute(permute)
            rho1 = rho1.permute(permute)
            pe_rpe_image_pairs.append(PePair(rho0, rho1))
            pe_rpe_permute_back.append(permute_back)
            pe_rpe_permute.append(permute)

        # Convert relative matrices to torch tensors
        pe_rpe_rel_mats = [torch.tensor(mat, dtype=dtype, device=device) for mat in pe_rpe_rel_mats]

        return pe_rpe_image_pairs, omega, m, h, pe_rpe_permute, pe_rpe_permute_back, pe_rpe_rel_mats
   


def create_normalized_grid(size, device, dtype):
    # size: tuple of (D, H, W) or (Z, Y, X) in permuted form
    z = torch.linspace(-1, 1, size[0], device=device, dtype=dtype)
    y = torch.linspace(-1, 1, size[1], device=device, dtype=dtype)
    x = torch.linspace(-1, 1, size[2], device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack((xx, yy, zz), dim=-1)  # shape: (D, H, W, 3)
    return grid        


def main():
    parser = argparse.ArgumentParser(description="PyHySCO: EPI-MRI Distortion Correction.")
    parser.add_argument("image_config", type=str, help="Path to the input 1 data file (NIfTI format .nii.gz)")
    parser.add_argument("--output_dir", type=str, default=" ", help="Directory to save the corrected images and reports (default=cwd)")
    parser.add_argument("--alpha", type=float, default=300, help="Smoothness regularization parameter (default=300)")
    parser.add_argument("--beta", type=float, default=1e-4, help="Intensity modulation constraint parameter (default=1e-4)")
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
    data = MultiPeDataObject(args.image_config, device=device, dtype=dtype, pair_idx=pair_idx, first_vol_only=True)


    # loss_func = EPIMRIDistortionCorrection(data, alpha=args.alpha, beta=args.beta, averaging_operator=args.averaging, derivative_operator=args.derivative, regularizer=args.regularizer, rho=args.rho, PC=args.PC)
    # B0 = loss_func.initialize(blur_result=True)
    # # set-up the optimizer
    # # change path to be where you want logfile and corrected images to be stored
    # opt = args.optimizer(loss_func, max_iter=args.max_iter, verbose=True, path=args.output_dir)
    # opt.run_correction(B0)
    # opt.apply_correction(method=args.correction)
    # if args.verbose:
    #     opt.visualize()

    initialization = InitializeCFMultiPeSimple()
    B0 = initialization.eval(data, blur_result=True)

    B0_detached = B0.detach()

    save_data(B0_detached.reshape(list(m_plus(data.m))).permute(data.permute_back[0]),
                os.path.join(args.output_dir, 'EstFieldMap.nii.gz'))


    rel_mat = data.rel_mats[1]

    pair_idx = [0, 1, 2, 3]
    data = MultiPeDataObject(args.image_config, device=device, dtype=dtype, pair_idx=pair_idx)
    # data.rel_mats[0] = rel_mat

    avg_operator = myAvg1D(data.omega, data.m, dtype=dtype, device=device)
    opt = LeastSquaresCorrectionMultiPeSparse4dMasked(data, avg_operator)
    corrected_image = opt.apply_correction(B0)
    save_data(corrected_image.reshape(list(data.m)).permute(data.permute_back[0]),
                os.path.join(args.output_dir, 'imCorrected.nii.gz'))


if __name__ == "__main__":
    main()
