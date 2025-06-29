from EPI_MRI.utils import *
from abc import ABC, abstractmethod
from EPI_MRI.LinearOperators import FFT3D, getLaplacianStencil
from torch.nn import functional as F
from EPI_MRI.utils import *


class InitializationMethod(ABC):
    """
    Defines initialization method to be used to initialize field map estimate.

    All children must implement an initialization evaluation method.
    """
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, data, *args, **kwargs):
        """
        Evaluates initialization.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        args, kwargs : Any
            Particular arguments and keyword arguments for initialization method.

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial guess for field map.
        """
        pass


class InitializeCF(InitializationMethod):
    """
    Defines parallelized one-dimensional initialization scheme from Chang & Fitzpatrick using optimal transport.
    """
    def __init__(self):
        super().__init__()

    def eval(self, data, blur_result=True, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        blur_result : boolean, optional
            Flag to apply Gaussian blur to `init_CF` result before returning (default is True).
        args, kwargs : Any
            Provided shift, if given (see method `init_CF`).

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial field map.
        """
        if blur_result:
            return self.blur(self.init_CF(data, *args, **kwargs).reshape(list(m_plus(data.m))), data.omega, data.m)
        else:
            return self.init_CF(data, *args, **kwargs)

    def init_CF(self, data, shift=2):
        """
        Optimal Transport based initialization scheme; an implementation of Chang & Fitzpatrick correction.

        Performs parallel 1-D optimal transport in distortion dimension to estimate field map.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        shift : float, optional
            Numeric shift to ensure smoothness of positive measure.

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        device = data.device
        dtype = data.dtype

        if torch.numel(data.m) == 3:
            rho0 = data.I1.data.reshape(-1, data.m[2])
            rho1 = data.I2.data.reshape(-1, data.m[2])
        elif torch.numel(data.m) == 4:
            rho0 = data.I1.data.reshape(-1, data.m[3])
            rho1 = data.I2.data.reshape(-1, data.m[3])
        else:
            rho0 = data.I1.data
            rho1 = data.I2.data

        rho0new = torch.empty(rho0.shape, device=device, dtype=dtype)
        rho1new = torch.empty(rho1.shape, device=device, dtype=dtype)
        rho0new = torch.add(rho0, shift, out=rho0new)
        rho1new = torch.add(rho1, shift, out=rho1new)

        rho0new = torch.div(rho0new, torch.sum(rho0new, dim=1, keepdim=True), out=rho0new)
        rho1new = torch.div(rho1new, torch.sum(rho1new, dim=1, keepdim=True), out=rho1new)

        C0 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device), torch.cumsum(rho0new, dim=1)),
            dim=1)
        C1 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device), torch.cumsum(rho1new, dim=1)),
            dim=1)
        C0[:, -1] = torch.ones_like(C0[:, 1])
        C1[:, -1] = torch.ones_like(C1[:, 1])

        t = torch.linspace(0, 1, int(data.m[-1] + 1), dtype=dtype, device=device).view(1, -1).expand(
            int(torch.prod(data.m) / data.m[-1]), -1)

        # interpolations

        iC0 = interp_parallel(C0, t, t, device=device)
        iC1 = interp_parallel(C1, t, t, device=device)

        iChf = torch.empty(iC0.shape, device=device)
        iChf = torch.div(torch.add(iC0, iC1, out=iChf), 2, out=iChf)

        T0hf = interp_parallel(t, iChf, C0, device=device)
        T1hf = interp_parallel(t, iChf, C1, device=device)

        T0hf = interp_parallel(T0hf, t, t, device=device)  # invert the mapping
        T1hf = interp_parallel(T1hf, t, t, device=device)  # invert the mapping

        T0hf = (data.omega[-2] - data.omega[-1]) * (T0hf - t)
        T1hf = (data.omega[-2] - data.omega[-1]) * (T1hf - t)

        Bc = torch.reshape(0.5 * (T0hf - T1hf), list(m_plus(data.m)))

        return -1 * Bc

    def blur(self, input, omega, m, alpha=1.0):
        """
        Performs Gaussian blur to pre-smooth initial field map.

        Parameters
        ----------
        input : torch.Tensor (size m_plus(m))
            Field map from `init_OT`.
        omega : torch.Tensor
            Image domain.
        m : torch.Tensor
            Image size.
        alpha : float, optional
            Standard deviation of Gaussian kernel (default is 1.0).

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map with blur.
        """
        L,_,_,_ = getLaplacianStencil(omega, m, input.dtype, input.device)
        K = FFT3D(L, m)
        return K.inv(input,1/alpha)


class InitializeRandom(InitializationMethod):
    """
    Defines random initialization scheme.
    """
    def __init__(self):
        super().__init__()

    def eval(self, data, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        args, kwargs : Any
            Provided seed, if given (see `rand_init`).

        Returns
        ----------
        B0 : torch.Tensor (size (m_plus(m),1))
            Initial field map.
        """
        return self.rand_init(data, *args, **kwargs)

    def rand_init(self, data, seed=None):
        """
        Random initialization scheme.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        seed : int, optional
            Seed for torch.random (for reproducibility) (default is None).

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(list(m_plus(data.m)), device=data.device, dtype=data.dtype)


class InitializeZeros(InitializationMethod):
    """
    Defines zero initialization scheme.
    """
    def __init__(self):
        super().__init__()

    def eval(self, data, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        args, kwargs : Any
            None for this initialization scheme.

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial field map.
        """
        return self.zero_init(data)

    def zero_init(self, data):
        """
        Zeros initialization scheme.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        return torch.zeros(list(m_plus(data.m)), device=data.device, dtype=data.dtype)


class InitializeCFMultiPe(InitializationMethod):
    """
    Defines parallelized one-dimensional initialization scheme from Chang & Fitzpatrick using optimal transport
    for multiple PE-RPE pairs.
    """
    def __init__(self):
        super().__init__()

    def eval(self, data, blur_result=True, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        blur_result : boolean, optional
            Flag to apply Gaussian blur to `init_CF` result before returning (default is True).
        args, kwargs : Any
            Provided shift, if given (see method `init_CF`).

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial field map.
        """
        if blur_result:
            return self.blur(self.init_CF(data, *args, **kwargs).reshape(list(m_plus(data.m))), data.omega, data.m)
        else:
            return self.init_CF(data, *args, **kwargs)

    def init_CF(self, data, shift=2):
        """
        Optimal Transport based initialization scheme; an implementation of Chang & Fitzpatrick correction
        for multiple PE-RPE pairs.

        Performs parallel 1-D optimal transport in distortion dimension to estimate field map for each pair,
        then averages the results.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        shift : float, optional
            Numeric shift to ensure smoothness of positive measure.

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        device = data.device
        dtype = data.dtype

        # Initialize list to store field maps from each pair
        field_maps = []

        # Process each PE-RPE pair
        for i, pair in enumerate(data.image_pairs):
            if torch.numel(data.m) == 3:
                rho0 = pair.pe_image.reshape(-1, data.m[2])
                rho1 = pair.rpe_image.reshape(-1, data.m[2])
            elif torch.numel(data.m) == 4:
                rho0 = pair.pe_image.reshape(-1, data.m[3])
                rho1 = pair.rpe_image.reshape(-1, data.m[3])
            else:
                rho0 = pair.pe_image
                rho1 = pair.rpe_image

            # Apply relative transformation matrix to RPE image coordinates
            if i > 0:  # Skip first pair as it's the reference
                # Create normalized coordinate grid for 2D slices
                coords = torch.stack(torch.meshgrid(
                    torch.linspace(-1, 1, data.m[0], device=device, dtype=dtype),
                    torch.linspace(-1, 1, data.m[1], device=device, dtype=dtype),
                    indexing='ij'
                ))  # Shape: (2, m0, m1)
                
                # Reshape for matrix multiplication
                coords_flat = coords.reshape(2, -1)  # Shape: (2, N)
                
                # Apply transformation to coordinates (only x,y components)
                coords_transformed = torch.matmul(data.rel_mats[i][:2, :2], coords_flat)
                
                # Reshape back to original shape
                coords_transformed = coords_transformed.reshape(2, data.m[0], data.m[1])
                
                # Permute for grid_sample (needs to be [N, H, W, 2])
                coords_transformed = coords_transformed.permute(1, 2, 0).unsqueeze(0)
                
                # Process each slice
                rho1_transformed = []
                for z in range(data.m[2]):
                    # Get the current slice and reshape it to 2D
                    if torch.numel(data.m) == 3:
                        slice_2d = rho1[:, z].reshape(data.m[0], data.m[1])
                    else:
                        slice_2d = rho1.reshape(data.m[0], data.m[1])
                    
                    # Interpolate image values at transformed coordinates for this slice
                    slice_transformed = F.grid_sample(
                        slice_2d.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                        coords_transformed,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=True
                    ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
                    rho1_transformed.append(slice_transformed)
                
                # Stack slices back together
                rho1 = torch.stack(rho1_transformed, dim=1)  # Stack along the second dimension

            rho0new = torch.empty(rho0.shape, device=device, dtype=dtype)
            rho1new = torch.empty(rho1.shape, device=device, dtype=dtype)
            rho0new = torch.add(rho0, shift, out=rho0new)
            rho1new = torch.add(rho1, shift, out=rho1new)

            rho0new = torch.div(rho0new, torch.sum(rho0new, dim=1, keepdim=True), out=rho0new)
            rho1new = torch.div(rho1new, torch.sum(rho1new, dim=1, keepdim=True), out=rho1new)

            C0 = torch.cat(
                (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device), torch.cumsum(rho0new, dim=1)),
                dim=1)
            C1 = torch.cat(
                (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device), torch.cumsum(rho1new, dim=1)),
                dim=1)
            C0[:, -1] = torch.ones_like(C0[:, 1])
            C1[:, -1] = torch.ones_like(C1[:, 1])

            t = torch.linspace(0, 1, int(data.m[-1] + 1), dtype=dtype, device=device).view(1, -1).expand(
                int(torch.prod(data.m) / data.m[-1]), -1)

            # interpolations
            iC0 = interp_parallel(C0, t, t, device=device)
            iC1 = interp_parallel(C1, t, t, device=device)

            iChf = torch.empty(iC0.shape, device=device)
            iChf = torch.div(torch.add(iC0, iC1, out=iChf), 2, out=iChf)

            T0hf = interp_parallel(t, iChf, C0, device=device)
            T1hf = interp_parallel(t, iChf, C1, device=device)

            T0hf = interp_parallel(T0hf, t, t, device=device)  # invert the mapping
            T1hf = interp_parallel(T1hf, t, t, device=device)  # invert the mapping

            T0hf = (data.omega[-2] - data.omega[-1]) * (T0hf - t)
            T1hf = (data.omega[-2] - data.omega[-1]) * (T1hf - t)

            # Store field map for this pair
            field_maps.append(torch.reshape(0.5 * (T0hf - T1hf), list(m_plus(data.m))))

        # Average field maps from all pairs
        Bc = torch.mean(torch.stack(field_maps), dim=0)

        return -1 * Bc

    def blur(self, input, omega, m, alpha=1.0):
        """
        Performs Gaussian blur to pre-smooth initial field map.

        Parameters
        ----------
        input : torch.Tensor (size m_plus(m))
            Field map from `init_OT`.
        omega : torch.Tensor
            Image domain.
        m : torch.Tensor
            Image size.
        alpha : float, optional
            Standard deviation of Gaussian kernel (default is 1.0).

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map with blur.
        """
        L,_,_,_ = getLaplacianStencil(omega, m, input.dtype, input.device)
        K = FFT3D(L, m)
        return K.inv(input,1/alpha)


class InitializeCFMultiPeSimple(InitializationMethod):
    """
    A simplified version of InitializeCFMultiPe that only uses the first PE-RPE pair.
    This avoids the complexity of handling multiple transformations.
    """
    def __init__(self):
        super().__init__()

    def eval(self, data, blur_result=True, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        blur_result : boolean, optional
            Flag to apply Gaussian blur to `init_CF` result before returning (default is True).
        args, kwargs : Any
            Provided shift, if given (see method `init_CF`).

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial field map.
        """
        if blur_result:
            return self.blur(self.init_CF(data, *args, **kwargs).reshape(list(m_plus(data.m))), data.omega, data.m)
        else:
            return self.init_CF(data, *args, **kwargs)

    def init_CF(self, data, shift=2):
        """
        Optimal Transport based initialization scheme using only the first PE-RPE pair.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        shift : float, optional
            Numeric shift to ensure smoothness of positive measure.

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        device = data.device
        dtype = data.dtype

        # Use only the first pair
        pair = data.image_pairs[0]
        
        if torch.numel(data.m) == 3:
            rho0 = pair.pe_image.reshape(-1, data.m[2])
            rho1 = pair.rpe_image.reshape(-1, data.m[2])
        elif torch.numel(data.m) == 4:
            rho0 = pair.pe_image.reshape(-1, data.m[3])
            rho1 = pair.rpe_image.reshape(-1, data.m[3])
        else:
            rho0 = pair.pe_image
            rho1 = pair.rpe_image

        rho0new = torch.empty(rho0.shape, device=device, dtype=dtype)
        rho1new = torch.empty(rho1.shape, device=device, dtype=dtype)
        rho0new = torch.add(rho0, shift, out=rho0new)
        rho1new = torch.add(rho1, shift, out=rho1new)

        rho0new = torch.div(rho0new, torch.sum(rho0new, dim=1, keepdim=True), out=rho0new)
        rho1new = torch.div(rho1new, torch.sum(rho1new, dim=1, keepdim=True), out=rho1new)

        C0 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device), torch.cumsum(rho0new, dim=1)),
            dim=1)
        C1 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device), torch.cumsum(rho1new, dim=1)),
            dim=1)
        C0[:, -1] = torch.ones_like(C0[:, 1])
        C1[:, -1] = torch.ones_like(C1[:, 1])

        t = torch.linspace(0, 1, int(data.m[-1] + 1), dtype=dtype, device=device).view(1, -1).expand(
            int(torch.prod(data.m) / data.m[-1]), -1)

        # interpolations
        iC0 = interp_parallel(C0, t, t, device=device)
        iC1 = interp_parallel(C1, t, t, device=device)

        iChf = torch.empty(iC0.shape, device=device)
        iChf = torch.div(torch.add(iC0, iC1, out=iChf), 2, out=iChf)

        T0hf = interp_parallel(t, iChf, C0, device=device)
        T1hf = interp_parallel(t, iChf, C1, device=device)

        T0hf = interp_parallel(T0hf, t, t, device=device)  # invert the mapping
        T1hf = interp_parallel(T1hf, t, t, device=device)  # invert the mapping

        T0hf = (data.omega[-2] - data.omega[-1]) * (T0hf - t)
        T1hf = (data.omega[-2] - data.omega[-1]) * (T1hf - t)

        Bc = torch.reshape(0.5 * (T0hf - T1hf), list(m_plus(data.m)))

        return -1 * Bc

    def blur(self, input, omega, m, alpha=1.0):
        """
        Performs Gaussian blur to pre-smooth initial field map.

        Parameters
        ----------
        input : torch.Tensor (size m_plus(m))
            Field map from `init_OT`.
        omega : torch.Tensor
            Image domain.
        m : torch.Tensor
            Image size.
        alpha : float, optional
            Standard deviation of Gaussian kernel (default is 1.0).

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map with blur.
        """
        L,_,_,_ = getLaplacianStencil(omega, m, input.dtype, input.device)
        K = FFT3D(L, m)
        return K.inv(input,1/alpha)


class InitializeCFMultiPe(InitializationMethod):
    """
    Defines parallelized one-dimensional initialization scheme from Chang & Fitzpatrick using optimal transport
    for multiple PE-RPE pairs.
    """

    def __init__(self):
        super().__init__()

    def eval(self, data, blur_result=True, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        blur_result : boolean, optional
            Flag to apply Gaussian blur to `init_CF` result before returning (default is True).
        args, kwargs : Any
            Provided shift, if given (see method `init_CF`).

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial field map.
        """
        if blur_result:
            return self.blur(self.init_CF(data, *args, **kwargs).reshape(
                list(m_plus(data.m))), data.omega, data.m)
        else:
            return self.init_CF(data, *args, **kwargs)

    def init_CF(self, data, shift=2):
        """
        Optimal Transport based initialization scheme; an implementation of Chang & Fitzpatrick correction
        for multiple PE-RPE pairs.

        Performs parallel 1-D optimal transport in distortion dimension to estimate field map for each pair,
        then averages the results.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        shift : float, optional
            Numeric shift to ensure smoothness of positive measure.

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        device = data.device
        dtype = data.dtype

        # Initialize list to store field maps from each pair
        field_maps = []

        # Process each PE-RPE pair
        for i, pair in enumerate(data.image_pairs):
            if torch.numel(data.m) == 3:
                rho0 = pair.pe_image.reshape(-1, data.m[2])
                rho1 = pair.rpe_image.reshape(-1, data.m[2])
            elif torch.numel(data.m) == 4:
                rho0 = pair.pe_image.reshape(-1, data.m[3])
                rho1 = pair.rpe_image.reshape(-1, data.m[3])
            else:
                rho0 = pair.pe_image
                rho1 = pair.rpe_image

            # Apply relative transformation matrix to RPE image coordinates
            if i > 0:  # Skip first pair as it's the reference
                # Create normalized coordinate grid for 2D slices
                coords = torch.stack(torch.meshgrid(
                    torch.linspace(-1, 1, data.m[0], device=device,
                                   dtype=dtype),
                    torch.linspace(-1, 1, data.m[1], device=device,
                                   dtype=dtype),
                    indexing='ij'
                ))  # Shape: (2, m0, m1)

                # Reshape for matrix multiplication
                coords_flat = coords.reshape(2, -1)  # Shape: (2, N)

                # Apply transformation to coordinates (only x,y components)
                coords_transformed = torch.matmul(data.rel_mats[i][:2, :2],
                                                  coords_flat)

                # Reshape back to original shape
                coords_transformed = coords_transformed.reshape(2, data.m[0],
                                                                data.m[1])

                # Permute for grid_sample (needs to be [N, H, W, 2])
                coords_transformed = coords_transformed.permute(1, 2,
                                                                0).unsqueeze(0)

                # Process each slice
                rho1_transformed = []
                for z in range(data.m[2]):
                    # Get the current slice and reshape it to 2D
                    if torch.numel(data.m) == 3:
                        slice_2d = rho1[:, z].reshape(data.m[0], data.m[1])
                    else:
                        slice_2d = rho1.reshape(data.m[0], data.m[1])

                    # Interpolate image values at transformed coordinates for this slice
                    slice_transformed = F.grid_sample(
                        slice_2d.unsqueeze(0).unsqueeze(0),
                        # Add batch and channel dimensions
                        coords_transformed,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=True
                    ).squeeze(0).squeeze(
                        0)  # Remove batch and channel dimensions
                    rho1_transformed.append(slice_transformed)

                # Stack slices back together
                rho1 = torch.stack(rho1_transformed,
                                   dim=1)  # Stack along the second dimension

            rho0new = torch.empty(rho0.shape, device=device, dtype=dtype)
            rho1new = torch.empty(rho1.shape, device=device, dtype=dtype)
            rho0new = torch.add(rho0, shift, out=rho0new)
            rho1new = torch.add(rho1, shift, out=rho1new)

            rho0new = torch.div(rho0new,
                                torch.sum(rho0new, dim=1, keepdim=True),
                                out=rho0new)
            rho1new = torch.div(rho1new,
                                torch.sum(rho1new, dim=1, keepdim=True),
                                out=rho1new)

            C0 = torch.cat(
                (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device),
                 torch.cumsum(rho0new, dim=1)),
                dim=1)
            C1 = torch.cat(
                (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device),
                 torch.cumsum(rho1new, dim=1)),
                dim=1)
            C0[:, -1] = torch.ones_like(C0[:, 1])
            C1[:, -1] = torch.ones_like(C1[:, 1])

            t = torch.linspace(0, 1, int(data.m[-1] + 1), dtype=dtype,
                               device=device).view(1, -1).expand(
                int(torch.prod(data.m) / data.m[-1]), -1)

            # interpolations
            iC0 = interp_parallel(C0, t, t, device=device)
            iC1 = interp_parallel(C1, t, t, device=device)

            iChf = torch.empty(iC0.shape, device=device)
            iChf = torch.div(torch.add(iC0, iC1, out=iChf), 2, out=iChf)

            T0hf = interp_parallel(t, iChf, C0, device=device)
            T1hf = interp_parallel(t, iChf, C1, device=device)

            T0hf = interp_parallel(T0hf, t, t,
                                   device=device)  # invert the mapping
            T1hf = interp_parallel(T1hf, t, t,
                                   device=device)  # invert the mapping

            T0hf = (data.omega[-2] - data.omega[-1]) * (T0hf - t)
            T1hf = (data.omega[-2] - data.omega[-1]) * (T1hf - t)

            # Store field map for this pair
            field_maps.append(
                torch.reshape(0.5 * (T0hf - T1hf), list(m_plus(data.m))))

        # Average field maps from all pairs
        Bc = torch.mean(torch.stack(field_maps), dim=0)

        return -1 * Bc

    def blur(self, input, omega, m, alpha=1.0):
        """
        Performs Gaussian blur to pre-smooth initial field map.

        Parameters
        ----------
        input : torch.Tensor (size m_plus(m))
            Field map from `init_OT`.
        omega : torch.Tensor
            Image domain.
        m : torch.Tensor
            Image size.
        alpha : float, optional
            Standard deviation of Gaussian kernel (default is 1.0).

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map with blur.
        """
        L, _, _, _ = getLaplacianStencil(omega, m, input.dtype, input.device)
        K = FFT3D(L, m)
        return K.inv(input, 1 / alpha)


class InitializeCFMultiePeDtiData(InitializationMethod):
    """
    A simplified version of InitializeCFMultiPe that only uses the first PE-RPE pair.
    This avoids the complexity of handling multiple transformations.
    """

    def __init__(self):
        super().__init__()

    def eval(self, data, blur_result=True, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        blur_result : boolean, optional
            Flag to apply Gaussian blur to `init_CF` result before returning (default is True).
        args, kwargs : Any
            Provided shift, if given (see method `init_CF`).

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial field map.
        """
        if blur_result:
            return self.blur(self.init_CF(data, *args, **kwargs).reshape(
                list(m_plus(data.m[1:]))), data.omega[2:], data.m[1:])
        else:
            return self.init_CF(data, *args, **kwargs)

    def init_CF(self, data, shift=2):
        """
        Optimal Transport based initialization scheme using only the first PE-RPE pair.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        shift : float, optional
            Numeric shift to ensure smoothness of positive measure.

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        device = data.device
        dtype = data.dtype

        # Use only the first pair
        pair = data.image_pairs[0]

        dwi_series_pe = pair[0]
        dwi_series_rpe = pair[1]

        mean_b0_pe = dwi_series_pe.data[dwi_series_pe.bval < 50].mean(0)
        mean_b0_rpe = dwi_series_rpe.data[dwi_series_rpe.bval < 50].mean(0)

        rho0 = mean_b0_pe.reshape(-1, mean_b0_pe.shape[-1])
        rho1 = mean_b0_rpe.reshape(-1, mean_b0_rpe.shape[-1])

        rho0new = torch.empty(rho0.shape, device=device, dtype=dtype)
        rho1new = torch.empty(rho1.shape, device=device, dtype=dtype)
        rho0new = torch.add(rho0, shift, out=rho0new)
        rho1new = torch.add(rho1, shift, out=rho1new)

        rho0new = torch.div(rho0new, torch.sum(rho0new, dim=1, keepdim=True),
                            out=rho0new)
        rho1new = torch.div(rho1new, torch.sum(rho1new, dim=1, keepdim=True),
                            out=rho1new)

        C0 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device),
             torch.cumsum(rho0new, dim=1)),
            dim=1)
        C1 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device),
             torch.cumsum(rho1new, dim=1)),
            dim=1)
        C0[:, -1] = torch.ones_like(C0[:, 1])
        C1[:, -1] = torch.ones_like(C1[:, 1])

        t = torch.linspace(0, 1, int(data.m[-1] + 1), dtype=dtype,
                           device=device).view(1, -1).expand(
            int(torch.prod(data.m[1:]) / data.m[-1]), -1)

        # interpolations
        iC0 = interp_parallel(C0, t, t, device=device)
        iC1 = interp_parallel(C1, t, t, device=device)

        iChf = torch.empty(iC0.shape, device=device)
        iChf = torch.div(torch.add(iC0, iC1, out=iChf), 2, out=iChf)

        T0hf = interp_parallel(t, iChf, C0, device=device)
        T1hf = interp_parallel(t, iChf, C1, device=device)

        T0hf = interp_parallel(T0hf, t, t, device=device)  # invert the mapping
        T1hf = interp_parallel(T1hf, t, t, device=device)  # invert the mapping

        T0hf = (data.omega[-2] - data.omega[-1]) * (T0hf - t)
        T1hf = (data.omega[-2] - data.omega[-1]) * (T1hf - t)

        Bc = torch.reshape(0.5 * (T0hf - T1hf), list(m_plus(data.m[1:])))

        return -1 * Bc

    def blur(self, input, omega, m, alpha=1.0):
        """
        Performs Gaussian blur to pre-smooth initial field map.

        Parameters
        ----------
        input : torch.Tensor (size m_plus(m))
            Field map from `init_OT`.
        omega : torch.Tensor
            Image domain.
        m : torch.Tensor
            Image size.
        alpha : float, optional
            Standard deviation of Gaussian kernel (default is 1.0).

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map with blur.
        """
        L, _, _, _ = getLaplacianStencil(omega, m, input.dtype, input.device)
        K = FFT3D(L, m)
        return K.inv(input, 1 / alpha)

class InitializeCFMultiePeDtiDataResampled(InitializationMethod):
    """
    A simplified version of InitializeCFMultiPe that only uses the first PE-RPE pair.
    This avoids the complexity of handling multiple transformations.
    """

    def __init__(self):
        super().__init__()

    def eval(self, data, target_res, blur_result=True, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        blur_result : boolean, optional
            Flag to apply Gaussian blur to `init_CF` result before returning (default is True).
        args, kwargs : Any
            Provided shift, if given (see method `init_CF`).

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial field map.
        """

        x = 1

        if blur_result:
            return self.blur(self.init_CF(data, target_res, *args, **kwargs).reshape(
                list(m_plus(target_res[1:]))), data.omega[2:], target_res[1:])
        else:
            return self.init_CF(data, target_res, *args, **kwargs)

    def init_CF(self, data, target_res, shift=2):
        """
        Optimal Transport based initialization scheme using only the first PE-RPE pair.

        Parameters
        ----------
        data : `MultiPeDataObject`
            Original image data with multiple PE-RPE pairs.
        shift : float, optional
            Numeric shift to ensure smoothness of positive measure.

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        device = data.device
        dtype = data.dtype

        # Use only the first pair
        pair = data.image_pairs[0]

        dwi_series_pe = pair[0]
        dwi_series_rpe = pair[1]

        mean_b0_pe = dwi_series_pe.data[dwi_series_pe.bval < 50].mean(0)
        mean_b0_rpe = dwi_series_rpe.data[dwi_series_rpe.bval < 50].mean(0)

        # Resample

        mean_b0_pe_resampled = F.interpolate(mean_b0_pe.unsqueeze(0), size=target_res[-2:].tolist(), mode='bilinear', align_corners=False).squeeze(0)
        mean_b0_rpe_resampled = F.interpolate(mean_b0_rpe.unsqueeze(0), size=target_res[-2:].tolist(), mode='bilinear', align_corners=False).squeeze(0)

        rho0 = mean_b0_pe_resampled.reshape(-1, mean_b0_pe_resampled.shape[-1])
        rho1 = mean_b0_rpe_resampled.reshape(-1, mean_b0_pe_resampled.shape[-1])

        rho0new = torch.empty(rho0.shape, device=device, dtype=dtype)
        rho1new = torch.empty(rho1.shape, device=device, dtype=dtype)
        rho0new = torch.add(rho0, shift, out=rho0new)
        rho1new = torch.add(rho1, shift, out=rho1new)

        rho0new = torch.div(rho0new,
                            torch.sum(rho0new, dim=1, keepdim=True),
                            out=rho0new)
        rho1new = torch.div(rho1new,
                            torch.sum(rho1new, dim=1, keepdim=True),
                            out=rho1new)

        C0 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device),
             torch.cumsum(rho0new, dim=1)),
            dim=1)
        C1 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device),
             torch.cumsum(rho1new, dim=1)),
            dim=1)
        C0[:, -1] = torch.ones_like(C0[:, 1])
        C1[:, -1] = torch.ones_like(C1[:, 1])

        t = torch.linspace(0, 1, int(target_res[-1] + 1), dtype=dtype,
                           device=device).view(1, -1).expand(
            int(torch.prod(target_res[1:]) / target_res[-1]), -1)

        # interpolations
        iC0 = interp_parallel(C0, t, t, device=device)
        iC1 = interp_parallel(C1, t, t, device=device)

        iChf = torch.empty(iC0.shape, device=device)
        iChf = torch.div(torch.add(iC0, iC1, out=iChf), 2, out=iChf)

        T0hf = interp_parallel(t, iChf, C0, device=device)
        T1hf = interp_parallel(t, iChf, C1, device=device)

        T0hf = interp_parallel(T0hf, t, t,
                               device=device)  # invert the mapping
        T1hf = interp_parallel(T1hf, t, t,
                               device=device)  # invert the mapping

        T0hf = (data.omega[-2] - data.omega[-1]) * (T0hf - t)
        T1hf = (data.omega[-2] - data.omega[-1]) * (T1hf - t)

        Bc = torch.reshape(0.5 * (T0hf - T1hf), list(m_plus(target_res[1:])))

        return -1 * Bc

    def blur(self, input, omega, m, alpha=1.0):
        """
        Performs Gaussian blur to pre-smooth initial field map.

        Parameters
        ----------
        input : torch.Tensor (size m_plus(m))
            Field map from `init_OT`.
        omega : torch.Tensor
            Image domain.
        m : torch.Tensor
            Image size.
        alpha : float, optional
            Standard deviation of Gaussian kernel (default is 1.0).

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map with blur.
        """
        L, _, _, _ = getLaplacianStencil(omega, m, input.dtype,
                                         input.device)
        K = FFT3D(L, m)
        return K.inv(input, 1 / alpha)