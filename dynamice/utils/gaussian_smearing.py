import numpy as np
import torch 

def gaussian_smearing(distances, offset, widths, centered=False):
    r"""Smear interatomic distance values using Gaussian functions.

    Args:
        distances (torch.Tensor): interatomic distances of (N_b x N_at x N_nbh) shape.
        offset (torch.Tensor): offsets values of Gaussian functions.
        widths: width values of Gaussian functions.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).

    Returns:
        torch.Tensor: smeared distances (N_b x N_at x N_nbh x N_g).

    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / widths**2
        # torch (grad) compatible expansion of indices
        offset = offset.reshape(1, 1, -1)
        if isinstance(distances, torch.Tensor):
            offset = torch.tensor(offset, device=distances.device)
            offset = offset.repeat(distances.shape[0], distances.shape[1], 1)
            diff = distances.unsqueeze(-1).expand((-1, -1, offset.shape[-1])) - offset
        else:
            diff = distances[:, :, None] - offset

    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / offset**2
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances.tile((1 ,1, offset.shape[0]))
    # compute smear distance values
    if isinstance(distances, torch.Tensor):
        coeff = torch.tensor(coeff, dtype=torch.float32, device=distances.device)
        return torch.exp(coeff*diff**2)

    gauss = np.exp(coeff * np.power(diff, 2))
    return gauss


class GaussianSmearing(object):
    r"""Smear layer using a set of Gaussian functions.

    Parameters
    ----------
    start: (float, optional)
        center of first Gaussian function, :math:`\mu_0`.
    stop: (float, optional)
        center of last Gaussian function, :math:`\mu_{N_g}`
    n_gaussians: (int, optional)
        total number of Gaussian functions, :math:`N_g`.

    width_factor: float, optional (default: 1.5)
        adjust the SD of gaussians.
        this is a constant factor multiplied by the bin width

    centered: (bool, optional)
        If True, Gaussians are centered at the origin and
        the offsets are used to as their widths (used e.g. for angular functions).
    margin: int
        The margin helps with the symmetric labels like angles. The margin specifies the
        number of bins to transfer to the head/tail of the bins from the other end.
        if zero, it will be skipped.

    normalize: bool, optional (default: False)
        if normalize final output of gaussians (divide by sum)

    """

    def __init__(
            self, start=0.0, stop=5.0, n_gaussians=50, margin=0,
            width_factor=1.5, centered=False, normalize=False
    ):

        # add margin
        self.margin = margin
        if margin > 0 :
            extra_domain = (stop - start)/n_gaussians * margin
            # self.upper_limit = stop - extra_domain
            # self.lower_limit = start + extra_domain
            start -= extra_domain
            stop += extra_domain
            n_gaussians += int(2*margin)


        # compute offset and width of Gaussian functions
        offsets = np.linspace(start, stop, n_gaussians, endpoint=False) #cyclic
        widths = width_factor * (offsets[1] - offsets[0]) * np.ones_like(offsets)

        self.offsets = offsets
        self.widths = widths
        self.centered = centered
        self.normalize = normalize

    def forward(self, features):
        """Compute smeared-gaussian distance values.

        Parameters
        ----------
        features: ndarray
            raw feature values of (batch_size, n_features) shape.

        Returns
        -------
        np.ndarray: layer output of (batch_size, n_features, n_gaussians) shape.

        """
        x = gaussian_smearing(
            features, self.offsets, self.widths, centered=self.centered
        )

        if isinstance(features, torch.Tensor):
            if self.margin > 0:
                helper_right = x[:, :, -self.margin:] + x[:, :, self.margin:2*self.margin]
                helper_left = x[:, :, :self.margin] + x[:, :, -2*self.margin:-self.margin]
                x = torch.cat([helper_right, x[:, :, 2*self.margin:-2*self.margin],
                              helper_left], dim=-1)
                
            if self.normalize:
                x = x / torch.sum(x, axis=-1)[..., None]
            return x

        if self.margin > 0:
            # mask_right = features>= self.upper_limit
            x[:, :, self.margin:2*self.margin] += x[:, :, -self.margin:]

            # mask_left = features<= self.lower_limit
            x[:, :, -2*self.margin:-self.margin] += x[:, :, :self.margin]
            x = x[:, :, self.margin:-self.margin]
        if self.normalize:
            x = x / x.sum(axis=-1)[..., None]
        return x
