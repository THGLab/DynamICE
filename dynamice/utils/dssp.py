import pandas as pd
import numpy as np

'''
helper functions for processing secondary structure data
compatible with Stride and mdtraj outputs
'''

def count_ssp(ssp, nseq, ssp_pop, helix=False):
    has_helix = 'H' in ssp #or 'I' in ssp or 'G' in ssp
    if helix and not has_helix: return 0
    for res in range(nseq):
        #print(res)
        if ssp[res] in [' ', 'C']: pass
        elif ssp[res] == 'T': ssp_pop['turn'].append(res + 1)
        elif ssp[res] in ['B', 'b']: ssp_pop['bridge'].append(res + 1)
        elif ssp[res] == 'H': ssp_pop['helix'].append(res + 1)
        elif ssp[res] == 'G': ssp_pop['310-helix'].append(res + 1)
        elif ssp[res] == 'I': ssp_pop['pi-helix'].append(res + 1)
        elif ssp[res] == 'E': ssp_pop['strand'].append(res + 1)
        elif ssp[res] == 'S': ssp_pop['bend'].append(res + 1)
        elif ssp[res] == 'NA': print('non protein residue')
        else: raise Exception('Secondary structure type not recognized, ', ssp[res])
    return 1


def read_stride_ssp(filename, nseq, ssp_pop):
    with open(filename, 'r') as f:
        ssp = ''
        prev_seq_count = 0
        for line in f.readlines():
            if line[:3] == 'SEQ':
                seq_count = int(line[63 : 65]) 
                #print(seq_count)
            if line[:3] == 'STR':
                ssp += line[10 : seq_count + 10 - prev_seq_count]
                #print(ssp)
                if seq_count == nseq:
                    assert len(ssp) == nseq
                    break
                prev_seq_count = seq_count
    count_ssp(ssp, nseq, ssp_pop)
    
def ssp_per_residue(ssp_pop, nseq, nstruct):
    ssp_res = {}
    for key in ssp_pop:
        ssp_res[key] = []
        #ssp_pop[key].sort()
        for n in range(nseq):
            ssp_res[key].append((np.array(ssp_pop[key]) == n+1).sum()/nstruct)
    return pd.DataFrame(ssp_res)

# secondary structures assignment by torsion angles using definitions from
#http://folding.chemistry.msstate.edu/utils/pross.html

agl_regions = {( 180, 180): 'A', ( 180,-120): 'B', ( 180, -60): 'C',
               ( 180,   0): 'D', ( 180,  60): 'e', ( 180, 120): 'F',
               (-120, 180): 'G', (-120,-120): 'h', (-120, -60): 'I',
               (-120,   0): 'J', (-120,  60): 'K', (-120, 120): 'L',
               ( -60, 180): 'M', ( -60,-120): 'N', ( -60, -60): 'O',
               ( -60,   0): 'P', ( -60,  60): 'Q', ( -60, 120): 'R',
               (   0, 180): 'S', (   0,-120): 't', (   0, -60): 'U',
               (   0,   0): 'V', (   0,  60): 'W', (   0, 120): 'X',
               (  60, 180): 'm', (  60,-120): 'r', (  60, -60): 'q',
               (  60,   0): 'p', (  60,  60): 'o', (  60, 120): 'n',
               ( 120, 180): 'g', ( 120,-120): 'l', ( 120, -60): 'k',
               ( 120,   0): 'j', ( 120,  60): 'i', ( 120, 120): 'h',
               ( 180,-180): 'A', (-180,-180): 'A', (-180, 180): 'A',
               (-180,-120): 'B', (-180, -60): 'C', (-180,   0): 'D',
               (-180,  60): 'e', (-180, 120): 'F', (-120,-180): 'G',
               ( -60,-180): 'M', (   0,-180): 'S', (  60,-180): 'm',
               ( 120,-180): 'g'}

TURN = ['OO', 'OP', 'OJ', 'PO', 'PP', 'PJ', 'JO', 'JP', 'JJ', 'rO',
        'Mo', 'Mp', 'Mj', 'Ro', 'Rp', 'Rj', 'oo', 'op', 'oj', 'rP',
        'po', 'pp', 'pj', 'jo', 'jp', 'jj', 'mO', 'mP', 'mJ', 'rJ']
          
HELIX = ['O', 'P'] # 5+
SHEET = ['L', 'G', 'F', 'A', 'R', 'M'] # 3+


def nearest_bound(angle):
    angle = np.array(angle)
    return (np.rint(angle/60)*60).astype(int)
        
def find_consecutive(x, limit):
    if x is None: return None, None
    x = np.reshape(x, -1)
    starts = []
    ends = []
    seq = x.copy()
    while len(seq) >= limit:
        minus = seq - np.arange(seq[0], seq[0]+len(seq))
        num = np.sum(minus == 0)
        if num >= limit:
            starts.append(seq[0])
            ends.append(seq[num-1])
        seq = seq[num:]
    return starts, ends         

def assign_ss(phi, psi):
    # phis shape (seq, ) of a structure
    backbones = np.vstack((phi, psi)).T #shape: seq, 2
    assert backbones.shape[1] == 2
    bounds = nearest_bound(backbones)
    paired_bound = map(tuple, bounds)
    letter = [agl_regions[r] for r in paired_bound]

    # check for helix & sheet
    hidx = np.argwhere([l in HELIX for l in letter])
    start, end = find_consecutive(hidx, 4)
    for n in range(len(start)):
        letter[start[n]:end[n]+1] = 'H'*(end[n]+1 - start[n])
    sidx = np.argwhere([l in SHEET for l in letter])

    start, end = find_consecutive(sidx, 3)
    for n in range(len(start)):
        letter[start[n]:end[n]+1] = 'E'*(end[n]+1 - start[n])

    # checks turn
    # construct dipeptide pairs
    dipeptide = [letter[n]+letter[n+1] for n in range(len(letter)-1)]
    tidx = np.argwhere([d in TURN for d in dipeptide])
    for i in np.reshape(tidx, -1):
        letter[i:i+2] = 'T'*2

    # else replace with empty string
    letter = [' ' if l not in ['H', 'E', 'T'] else l for l in letter]
    assert len(letter) == len(phi)
    return ''.join(letter)


