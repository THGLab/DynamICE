"""
Nov 21  2021
@author: oufan

Extracting torsion angles and/or coordinates from pdbs
"""

import numpy as np
import pandas as pd
from dynamice.utils import GetTorsion, get_backbone_pdb
from idpconfgen.libs.libcalc import calc_torsion_angles
#from sklearn.model_selection import train_test_split
#import Bio.PDB as PDB
import os
import time
#from Bio.PDB.vectors import calc_angle, calc_dihedral

def padzeros(chis, name, seq):
    pad_idx = get_padding_idx(seq)
    chi = chis[name].values.reshape(-1, len(seq)-len(pad_idx))
    # insert space for GLY ALA torsions
    chi = np.insert(chi, pad_idx, np.nan, axis=1)
    assert chi.shape[-1] == len(seq)
    return chi

def get_padding_idx(seq):
    # get index for inserting nan values to seq without ALA GLY
    idx = [cres for cres in range(len(seq)) if seq[cres] in ['A', 'G']]
    offsets = np.arange(len(idx))
    return np.array(idx) - offsets

def get_chis(df, npdbs, seq):
    chis = np.zeros((5, npdbs, len(seq)))
    for n in range(5):
        chis[n] = padzeros(df, 'chi'+str(n+1), seq)
    return np.stack(chis, axis=-1)

def get_bbs(pdblist, nseq):
    bb_torsions = np.zeros((len(pdblist), (nseq-1)*3))
    nf = 0
    for file in pdblist:
        bb_generator = get_backbone_pdb(file, 1)
        bbcoords = next(bb_generator)[0]
        #print(bbcoords.shape)
        bb_torsions[nf] = calc_torsion_angles(bbcoords)
        nf += 1
        if nf%50 == 0: time.sleep(5)
    bb_torsions = np.degrees(bb_torsions)
    # insert first and last residue
    backbones = np.insert(bb_torsions, [0, 0, (nseq-1)*3], np.nan, axis=1)
    return backbones


# sidechain outfile
idp_path = 'torsions/asyn_mcsce'
sc_out = os.path.join(idp_path, 'sc_LH.out')
pdblist = ['torsions/asyn_mcsce/total/%i.pdb'%(n+1) for n in range(7625, 10085)]
sc = GetTorsion().read_torsions(pdblist, outfile=sc_out, chi=[1,2,3,4,5])

sequence = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'
#'MEAIAKHDFSATADDELSFRKTQILKILNMEDDSNWYRAELDGKEGLIPSNYIEMKNHD'

seq_len = len(sequence)
sc_read = pd.read_csv(sc_out)[['chi1', 'chi2', 'chi3', 'chi4', 'chi5']]
chis = get_chis(sc_read, len(pdblist), sequence)
backbones = get_bbs(pdblist, seq_len)
bbsc = np.dstack((np.reshape(backbones, (-1, seq_len, 3)), chis))
bbsc = bbsc.reshape(-1, seq_len*8)
np.save(os.path.join(idp_path, 'bbsc2.npy'), bbsc)
#save_torsion_npy(bbsc, 1500, 100, idp_path)


