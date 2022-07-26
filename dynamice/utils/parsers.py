"""
Miscellaneous utility functions
"""

import numpy as np
from Bio.PDB import PDBParser

"""
Copyright: https://github.com/joaomcteixeira/IDPConformerGenerator/blob/cacb4be7effd51da6876ca242ed2250766198453/src/idpconfgen/libs/libcalc.py

"""

def calc_torsion_angles(
        coords,
        ARCTAN2=np.arctan2,
        CROSS=np.cross,
        DIAGONAL=np.diagonal,
        MATMUL=np.matmul,
        NORM=np.linalg.norm,
        ):
    """
    Calculate torsion angles from sequential coordinates.
    Uses ``NumPy`` to compute angles in a vectorized fashion.
    Sign of the torsion angle is also calculated.
    Uses Prof. Azevedo implementation:
    https://azevedolab.net/resources/dihedral_angle.pdf
    Example
    -------
    Given the sequential coords that represent a dummy molecule of
    four atoms:
    >>> xyz = numpy.array([
    >>>     [0.06360, -0.79573, 1.21644],
    >>>     [-0.47370, -0.10913, 0.77737],
    >>>     [-1.75288, -0.51877, 1.33236],
    >>>     [-2.29018, 0.16783, 0.89329],
    >>>     ])
    A1---A2
           \
            \
            A3---A4
    Calculates the torsion angle in A2-A3 that would place A4 in respect
    to the plane (A1, A2, A3).
    Likewise, for a chain of N atoms A1, ..., An, calculates the torsion
    angles in (A2, A3) to (An-2, An-1). (A1, A2) and (An-1, An) do not
    have torsion angles.
    If coords represent a protein backbone consisting of N, CA, and C
    atoms and starting at the N-terminal, the torsion angles are given
    by the following slices to the resulting array:
    - phi (N-CA), [2::3]
    - psi (CA-C), [::3]
    - omega (C-N), [1::3]
    Parameters
    ----------
    coords : numpy.ndarray of shape (N>=4, 3)
        Where `N` is the number of atoms, must be equal or above 4.
    Returns
    -------
    numpy.ndarray of shape (N - 3,)
        The torsion angles in radians.
        If you want to convert those to degrees just apply
        ``np.degrees`` to the returned result.
    """
    # requires
    assert coords.shape[0] > 3
    assert coords.shape[1] == 3

    crds = coords.T

    # Yes, I always write explicit array indices! :-)
    q_vecs = crds[:, 1:] - crds[:, :-1]
    cross = CROSS(q_vecs[:, :-1], q_vecs[:, 1:], axis=0)
    unitary = cross / NORM(cross, axis=0)

    # components
    # u0 comes handy to define because it fits u1
    u0 = unitary[:, :-1]

    # u1 is the unitary cross products of the second plane
    # that is the unitary q2xq3, obviously applied to the whole chain
    u1 = unitary[:, 1:]

    # u3 is the unitary of the bonds that have a torsion representation,
    # those are all but the first and the last
    u3 = q_vecs[:, 1:-1] / NORM(q_vecs[:, 1:-1], axis=0)

    # u2
    # there is no need to further select dimensions for u2, those have
    # been already sliced in u1 and u3.
    u2 = CROSS(u3, u1, axis=0)

    # calculating cos and sin of the torsion angle
    # here we need to use the .T and np.diagonal trick to achieve
    # broadcasting along the whole coords chain
    # np.matmul is preferred to np.dot in this case
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    cos_theta = DIAGONAL(MATMUL(u0.T, u1))
    sin_theta = DIAGONAL(MATMUL(u0.T, u2))

    # torsion angles
    return -ARCTAN2(sin_theta, cos_theta)


def get_backbone_pdb(path, pdb_split=1000):
    """
    reads a PDB file with several models and yields backbone structure.

    Parameters
    ----------
    path: str
        path to the pdb file

    pdb_split: int, optional (default: 1000)
        length of splits

    Yields
    -------
    ndarray: 3D array of backbone atoms of all pdb data with shape (L, A, 3),
        where L is the `split_length`, and A is the number of atoms.

    """

    # read energy file to dataframe
    parser = PDBParser()
    data = parser.get_structure('data', path)
    models = data.get_models()

    i_count = 0
    n_atoms = 0
    splits = []
    while True:
        try:
            model = next(models)
            i_count += 1


            atoms = []
            for residue in model.get_residues():
                atoms += [residue['N'].get_coord(), residue['CA'].get_coord(),
                          residue['C'].get_coord()]

            atoms = np.array(atoms)

            # sanity check
            if i_count == 1:
                n_atoms = atoms.shape[0]
            else:
                assert atoms.shape == (n_atoms, 3)

            splits.append(atoms)

            if i_count % pdb_split == 0:
                splits = np.array(splits)
                assert splits.shape == (pdb_split, n_atoms, 3)
                yield splits

                splits = []

        except StopIteration:
            break
