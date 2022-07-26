"""
Helper functions for training with distance restraints
"""

import torch
import numpy as np


def find_coordpair_idx(exp_data, atom_labels):
    """
    Parameters
    ----------
    exp_data: DataFrame
    atom_labels: ndarray of atom labels (N*24 including Hs & paddings) 
    Returns
    -------
    np.ndarray: indices of coordinate pairs whose distances are defined in data
    """
    
    indices = []
    ifmulti = 'atom1_multiple_assignments' in exp_data.columns.values
    for idx in range(exp_data.shape[0]):
        row = exp_data.iloc[idx]
        res1 = int(row.res1) - 1 
        res2 = int(row.res2) - 1
        multi1 =  0 if not ifmulti else row.atom1_multiple_assignments 
        multi2 =  0 if not ifmulti else row.atom2_multiple_assignments
        atom1 = row.atom1 if 'atom1' in exp_data.columns.values else 'CA'
        atom2 = row.atom2 if 'atom2' in exp_data.columns.values else 'CA'
        atom1_idxs = get_idx_from_label(atom1, multi1, atom_labels[res1])
        atom2_idxs = get_idx_from_label(atom2, multi2, atom_labels[res2])
        pairs = format_res_idx_pair(res1, atom1_idxs, res2, atom2_idxs)
        indices.append(pairs)

    return indices

def torch_vector_norm(x):
    # assumes 1-dim
    assert x.ndim == 1
    return torch.sqrt(torch.sum((x**2).unsqueeze(0), -1))

def calc_distances(coords, indices):
    # indices shape n_data, distance_pairs
    # compatible with torch
    bc = []
    iftorch = isinstance(coords, torch.Tensor)
    for datapoint in indices:
        dists = 0.
        nd = 0.
        for (res1, idx1, res2, idx2)  in datapoint:
            if iftorch: 
                dists += torch_vector_norm(coords[res1, idx1]-coords[res2, idx2])**(-6.)
            else:
                dists += np.linalg.norm(coords[res1, idx1]-coords[res2, idx2])**(-6.)
            nd += 1.
        dists = (dists/nd)**(-1./6.)
        bc.append(dists)
    return torch.cat(bc, dim=0) if iftorch else np.array(bc)
                

def format_res_idx_pair(res1, idx1, res2, idx2):
    indices = []
    nidx1 = len(idx1)
    nidx2 = len(idx2)
    for i in range(nidx1):
        for j in range(nidx2):
            indices.append((res1, idx1[i], res2, idx2[j]))
    return indices
    

def get_idx_from_label(atomname, multiple, labels):
    # labels needs to be ndarray
    idx = []
    if atomname == 'H':
        return np.argwhere(labels=='H')[0]
    elif atomname == 'OXT':
        return [23]
    # deals with incorrectly formatted atomnames
    for n in range(len(labels)):
        if atomname in labels[n]:
            idx.append(n)
        if not multiple and len(idx)==1:
            break
    return idx
   

def calc_jc(phis, idxs):
    #return torch.cos(phis[idxs] - 3.14159/3.)
    return karplus(phis[idxs])

def karplus(phis):
    sig = {'A': np.sqrt(0.14), 'B': np.sqrt(0.03), 'C':np.sqrt(0.08)}
    mu = {'A': 6.51, 'B': -1.76, 'C': 1.6}
    #A = np.random.normal(mu['A'], sig['A'])
    #B = np.random.normal(mu['B'], sig['B'])
    #C = np.random.normal(mu['C'], sig['C'])
    alphas = torch.cos(phis - 3.14159/3.)
    return mu['A']*alphas**2 + mu['B']*alphas + mu['C']



def calc_smfret(distance, r0):
    # 'CA to CA' distance
    if isinstance(distance, torch.Tensor):
        r0 = torch.tensor(r0, dtype=torch.float32, device=distance.device)
    return 1./(1. + (distance/r0)**6.)

