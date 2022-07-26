import numpy as np

# number of sidechain torsion for 20 amino acids
sidechain_res = {'A': 0, 'G': 0, 'S': 1, 'T': 1, 'V': 1, 'C': 1, 'I': 2,
                 'L': 2, 'P': 2, 'H': 2, 'W': 2, 'Y': 2, 'F': 2, 'D': 2,
                 'N': 2, 'E': 3, 'M': 3, 'Q': 3, 'K': 4, 'R': 5}

BOND_ANGLES  = np.array([2.124, 1.941, 2.028])
BOND_ANGLE_SIGMA  = np.array([0.0274, 0.0435, 0.0211])
BB_TORTYPE = ['CX-C -N -CX', 'C -N -CX-C ', 'N -CX-C -N ']

# https://www.cgl.ucsf.edu/chimerax/docs/user/radii.html
vdW_radii_tsai_1999 = {'C': 1.7, 'H': 1.0, 'N': 1.625, 'O': 1.480,
                       'P': 1.871, 'S': 1.782}

# assumes the atom labels follows standard pdb naming order (heavy 
# atoms first then H atoms)
heavy_atom_num = {'G': 4, 'A': 5, 'S': 6, 'T': 7, 'V': 7, 'C': 6, 'I': 8,
                  'L': 8, 'P': 7, 'H': 10, 'W': 14, 'Y': 12, 'F': 11, 'D': 8,
                  'N': 8, 'E': 9, 'M': 8, 'Q': 9, 'K': 9, 'R': 11, 'p':10,
                  'd': 10, 'e': 10}
# Warning: considers all Histidine as HIP 
Hatom_num = {'G': 3, 'A': 5, 'S': 5, 'T': 7, 'V': 9, 'C': 5, 'I': 11,
             'L': 11, 'P': 7, 'W': 10, 'Y': 9, 'F': 9, 'D': 4,
             'N': 6, 'E': 6, 'M': 9, 'Q': 8, 'K': 13, 'R': 13, 'H': 8}
