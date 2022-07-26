import numpy as np
import re
import torch
from dynamice.utils.definitions import (
  sidechain_res,
  BOND_ANGLES,
  BOND_ANGLE_SIGMA,
  heavy_atom_num,
  Hatom_num,
)

def generate_tor_marker(seq):
    # generates an array of torsion type encoders given a sequence
    marker = []
    for nres in range(len(seq)):
        chunk = np.arange(sidechain_res[seq[nres]] + 3)
        # remove psi from last res
        if nres == len(seq) - 1:
            chunk = np.concatenate((chunk[:2], chunk[3:]))
        marker += list(chunk)
    # remove omega, phi from 1st res
    return np.array(marker[2:])


def residue_sc_marker(seq):
    # generates an array of the number of sidechain torsions for a sequence
    return np.array([sidechain_res[res] for res in seq])

def torsion_align(tor1, tor2, seq):
    # align torsions for correlation plot, returns the shared idx (for tor1)
    tor_name = {'psi': 2, 'chi1': 3, 'chi2': 4, 'chi3': 5}
    tor1 = tor_name[tor1]
    tor2 = tor_name[tor2]
    res_num = np.array([sidechain_res[res] for res in seq])
    tor1_res = np.argwhere(res_num+2 >= tor1).flatten()
    if tor1 == 2: tor1_res = tor1_res[:-1]
    tor2_res = np.argwhere(res_num+2 >= tor2).flatten()
    share_idx = [n for n in range(len(tor1_res)) if tor1_res[n] in tor2_res]
    return share_idx


def get_heavy_coords_idx(seq):
    idx = []
    # terminal special atoms: OXT not included
    for cres in range(len(seq)):
        if cres == 0:
            idx += list(np.arange(heavy_atom_num[seq[cres]]))
            st = heavy_atom_num[seq[cres]] + Hatom_num[seq[cres]] + 2 #H2/3
        else:
            idx += list(st + np.arange(heavy_atom_num[seq[cres]]))
            st += heavy_atom_num[seq[cres]] + Hatom_num[seq[cres]]
    #idx.append(-1)
    return idx
          
def scn_coords_unpad(seq):
    # returns the indices of heavy atom coordinates from sidechainnet formatted coordinates (14xnum_res, 3)
    # OXT not included
    idx = []
    for cres in range(len(seq)):
        idx += list(np.arange(heavy_atom_num[seq[cres]])+ 14*cres)
    return idx

# numpy version of sidechainnet torsion to coordinate conversion
def get_scn_angles(torsions, res_mark, seq):
    # single structure angles for numpy
    if np.any(torsions > 10): torsions = np.radians(torsions)
    nseq = len(seq)
    angles = np.full((nseq, 12), -np.pi)
    
    for cres in range(nseq):
        # fill in torsions, order 0:3 phi, psi, omega, 7:12 chis
        # correct residues with alternative torsions
        if cres == 0:
            angles[cres, 1] = torsions[0]
            angles[cres, 2] = torsions[res_mark[cres]+1]
            angles[cres, 7:7+res_mark[cres]] = torsions[1:1+res_mark[cres]] 
        elif cres == nseq-1:
            angles[cres, 0] = torsions[-res_mark[cres]-1]
            if res_mark[cres] > 0:
                angles[cres, 7:7+res_mark[cres]] = torsions[-res_mark[cres]:]
        else:
            catom = np.sum(res_mark[:cres]) + cres*3 - 1
            angles[cres, 0:2] = torsions[catom:catom+2]
            angles[cres, 2] = torsions[catom+res_mark[cres]+2]
            angles[cres, 7:7+res_mark[cres]] = torsions[catom+2:catom+res_mark[cres]+2]
        if seq[cres] in ['V', 'L']:
            angles[cres, 7+res_mark[cres]] = cyclic_angle(angles[cres, 6+res_mark[cres]]+np.radians(120))
        elif seq[cres] == 'I':
            angles[cres, 7+res_mark[cres]] = cyclic_angle(angles[cres, 5+res_mark[cres]]-np.radians(120))
        # idx 6 C-N-CA-CB by phi - 120 degrees
        angles[cres, 6] = cyclic_angle(angles[cres, 0] - np.radians(120)) 
        # get bond angles
        angles[cres, 3:6] = np.random.normal(BOND_ANGLES, BOND_ANGLE_SIGMA)
    return angles

def get_scn_structure(torsions, sequence, hydrogen=False, save_pdb=None):
    # get single structure from torsions numpy version
    from sidechainnet.StructureBuilder import StructureBuilder
    struct = StructureBuilder(sequence, torsions)
    coord = struct.build()
    if hydrogen:
        struct.add_hydrogens()
        oxt = struct.terminal_atoms['OXT']
        coord = struct.coords
        # not compatible with all residues
        coord[-1, :] = oxt
    if save_pdb is not None:
        struct.to_pdb(save_pdb)
    return coord
    

## torch version of sidechainnet coordinates conversion
def cyclic_angle(tor):
    if isinstance(tor, torch.Tensor):
        for n in range(tor.shape[0]):
            a = tor[n] if tor[n] >= -3.14159 else 2*3.14159 + tor[n]
            tor[n] = a
    else:
        tor = tor if tor >= -np.pi else 2*np.pi + tor
    return tor

def get_batched_angles(torsions, res_mark, seq, device):
    nseq = len(res_mark)
    batch_size = torsions.shape[0]
    batch_angles = torch.full((batch_size, nseq, 12), -3.14159, 
                               dtype=torch.float32, device=device)
   
    for cres in range(nseq):
            # fill in torsions, order 0:3 phi, psi, omega, 7:12 chis
        if cres == 0:
            batch_angles[:, cres, 1] = torsions[:, 0]
            batch_angles[:, cres, 2] = torsions[:, res_mark[cres]+1]
            batch_angles[:, cres, 7:7+res_mark[cres]] = torsions[:, 1:1+res_mark[cres]] 
        elif cres == nseq-1:
            batch_angles[:, cres, 0] = torsions[:, -res_mark[cres]-1]
            if res_mark[cres] > 0:
                batch_angles[:, cres, 7:7+res_mark[cres]] = torsions[:, -res_mark[cres]:]
        else:
            catom = np.sum(res_mark[:cres]) + cres*3 - 1
            batch_angles[:, cres, 0:2] = torsions[:, catom:catom+2]
            batch_angles[:, cres, 2] = torsions[:, catom+res_mark[cres]+2]
            batch_angles[:, cres, 7:7+res_mark[cres]] = torsions[:, catom+2:catom+res_mark[cres]+2]
        if seq[cres] in ['V', 'L']:
            batch_angles[:, cres, 7+res_mark[cres]] = cyclic_angle(batch_angles[:, cres, 6+res_mark[cres]]+2*3.14159/3.)
        elif seq[cres] == 'I':
            batch_angles[:, cres, 7+res_mark[cres]] = cyclic_angle(batch_angles[:, cres, 5+res_mark[cres]]-2*3.14159/3.)
        # idx 6 C-N-CA-CB by phi - 120 degrees
        batch_angles[:, cres, 6] = cyclic_angle(batch_angles[:, cres, 0] - 2*3.14159/3.)
        # get bond angles
        ba = np.random.normal(BOND_ANGLES, BOND_ANGLE_SIGMA, (batch_size, 3))
        batch_angles[:, cres, 3:6] = torch.tensor(ba, dtype=torch.float32, device=device)
    return batch_angles


def get_batched_coords(torsions, sequence, res_mark, 
                       device=None, hydrogen=False, remove_pad=False):
    from sidechainnet.BatchedStructureBuilder import BatchedStructureBuilder as batchbuilder
    if device is None: device = torsions.device
    batch_agls = get_batched_angles(torsions, res_mark, sequence, device)
    batch_build = batchbuilder(sequence, device, batch_agls)
    all_coords = batch_build.build(hydrogen)
    if remove_pad:
        unpad_idx = scn_coords_unpad(sequence)
        all_coords = all_coords[:, unpad_idx, :]
    #print(np.where(np.all(np.isclose(all_coords.detach().numpy()[0], 0), axis=-1))[0])
    return all_coords, batch_agls


def get_scn_labels(sequence):
    from sidechainnet.sequence import ATOM_MAP_24
    labels = []
    for res in sequence:
        labels.append(ATOM_MAP_24[res])
    return np.array(labels)



# some helper functions for amino-categorized ramanchandran plot analysis
def get_unrepeated_res(seq):
    res_used = []
    for res in seq:
        if res not in res_used: res_used.append(res)
    return res_used

def get_res_idx(seq, reslist='G'):
    # res: list of 1-letter string or 1-letter string
    if isinstance(reslist, str):
        return [m.start() for m in re.finditer(reslist, seq)]
    match = []
    for res in reslist:
        match += [m.start() for m in re.finditer(res, seq)]
    return match

def get_pre_proline_idx(seq):
    match = re.finditer('P', seq)
    res_idx = [m.start() for m in match]
    prepro = [n-1 for n in res_idx if n > 0]
    return prepro

def get_proline_chis(seq):
    pro_idx = np.array(get_res_idx(seq, 'P'))
    offset1 = np.zeros(len(pro_idx), dtype=np.int8)
    offset2 = np.zeros(len(pro_idx), dtype=np.int8)
    for p in range(len(pro_idx)):
        for n in range(pro_idx[p]):
            if sidechain_res[seq[n]]<1: offset1[p] += 1 
            if sidechain_res[seq[n]]<2: offset2[p] += 1 
    pro_chi1_idx = pro_idx - offset1
    pro_chi2_idx = pro_idx - offset2
    return pro_chi1_idx, pro_chi2_idx

def get_other_idx(seq):
    prepro = get_pre_proline_idx(seq)
    spec_idx = prepro + get_res_idx(seq, ['G','P'])
    non_idx = [n for n in range(len(seq)) if n not in spec_idx]
    return non_idx

def get_bb_by_res(data, res_idx):
    # data (2*batch, sequence length) torsion order (phi, psi)
    # res 1-letter string
    res_bb = {}
    nbatch = int(data.shape[0]/2)
    phi = data[:nbatch, res_idx]
    psi = data[nbatch:, res_idx]
    res_bb['phi'] = phi
    res_bb['psi'] = psi
    return res_bb

    

def euler_rotation_matrix(theta):
    """
    Rotate the xyz values based on the euler angles, theta. Directly copied from:
    Credit:"https://www.learnopencv.com/rotation-matrix-to-euler-angles/"

    Parameters
    ----------
    theta: numpy array
        A 1D array of angles along x, y and z directions

    Returns
    -------
    numpy array: rotation matrix with shape (3,3)
    """

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def rotate_molecule(atoms, theta=None):
    """
    Rotates the structure of molecule between -pi/2 and pi/2.

    Parameters
    ----------
    atoms: numpy array
        An array of atomic positions with last dimension = 3

    theta: numpy array, optional (default: None)
        A 1D array of angles along x, y and z directions.
        If None, it will be generated uniformly between -pi/2 and pi/2

    Returns
    -------
    numpy array: The rotated atomic positions with shape (... , 3)

    """

    # handle theta
    if theta is None:
        theta = np.random.uniform(-np.pi / 2., np.pi / 2., size=3)

    # rotation matrix
    R = euler_rotation_matrix(theta)

    return np.dot(atoms, R)


def padaxis(array, new_size, axis, pad_value=0, pad_right=True):
    """
    Padds one axis of an array to a new size
    This is just a wrapper for np.pad, more usefull when only padding a single axis

    Parameters
    ----------
    array: ndarray
        the array to pad

    new_size: int
        the new size of the specified axis

    axis: int
        axis along which to pad

    pad_value: float or int, optional(default=0)
        pad value

    pad_right: bool, optional(default=True)
        if True pad on the right side, otherwise pad on left side

    Returns
    -------
    ndarray: padded array

    """
    add_size = new_size - array.shape[axis]
    assert add_size >= 0, 'Cannot pad dimension {0} of size {1} to smaller size {2}'.format(axis, array.shape[axis], new_size)
    pad_width = [(0,0)]*len(array.shape)

    #pad after if int is provided
    if pad_right:
        pad_width[axis] = (0, add_size)
    else:
        pad_width[axis] = (add_size, 0)

    return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)
