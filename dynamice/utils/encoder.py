import numpy as np


def onehot_digitizer(features, smearing, binary=True):
    """
    This function turns arrays of float values to the one hot vectors based on the
    specified number of bins in the given range of values.

    Parameters
    ----------
    features: ndarray
        The input array of (n_batch, n_values) shape.

    smearing: tuple
        (start, stop, n_bins)

    binary: bool, optional (default: False)
        if True, returns binary digits, otherwise returns index of class.

    Returns
    -------
    ndarray: The 3D numpy array of (n_batch, n_values, n_bins/1) shape.

    """
    if smearing is None:
        return features

    start, stop, n_bins = smearing
    offset = np.linspace(start, stop, n_bins, endpoint=True)
    diff = np.abs(features[:,:,None] - offset[None, None, :])

    digit = np.argmin(diff, axis=-1)  # n_batch, n_values

    if not binary:
        return digit
    else:
        ohe = np.zeros((digit.size, n_bins))
        ohe[np.arange(digit.size), digit.flatten()] = 1
        ohe = ohe.reshape(digit.shape[0], digit.shape[1], n_bins)

        return ohe

def onehot_digitizer_distance(features, smearing, binary=True):
    """
    This function turns arrays of float values to the one hot vectors based on the
    specified number of bins in the given range of values.

    Parameters
    ----------
    features: ndarray
        The input array of (n_batch, n_values) shape.

    smearing: tuple
        a tuple of 3 tuples, each with information about start, stop, and n_bins

    binary: bool, optional (default: False)
        if True, returns binary digits, otherwise returns index of class.

    Returns
    -------
    ndarray: The 3D numpy array of (n_batch, n_values, n_bins/1) shape.

    """
    if smearing is None:
        return features

    digits = np.zeros((features.shape[0], features.shape[1]))   # B,A
    ohe = np.zeros((features.shape[0], features.shape[1], smearing[0][2]))  # B,A,n_bins
    for i, smear in enumerate(smearing):
        n_bins = smear[2]
        offset = np.linspace(smear[0], smear[1], n_bins, endpoint=True)
        diff = np.abs(features[:, i::3, None] - offset[None, None, :])

        digit = np.argmin(diff, axis=-1)  # B, A/3

        #if not binary:
        digits[:,i::3] = digit

        #else:
        ohe_ = np.zeros((digit.size, n_bins))
        ohe_[np.arange(digit.size), digit.flatten()] = 1
        ohe_ = ohe_.reshape(digit.shape[0], digit.shape[1], n_bins)    # B, A/3, n_bins

        ohe[:,i::3,:] = ohe_

    if not binary:
        return digits
    else:
        return ohe
