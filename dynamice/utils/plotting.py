"""
Created on Tue Nov 30 2021
@author: oufan

Plotting tools for torsion analysis
"""

import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import numpy as np
import seaborn as sns
from dynamice.utils.utility import (torsion_align, sidechain_res, get_other_idx,
                            get_bb_by_res, get_pre_proline_idx, get_res_idx)

# torsion distribution plots
def torsion_hist(tor, torname, savedir=None):
    fig = plt.figure(figsize = (8, 6))
    plt.hist(tor, bins=np.arange(-180, 180, 5), density=True, color='yellowgreen', edgecolor='black') #'indianred' #steelblue
    if 'chi' in torname:
        torname = 'chi_' + torname[-1]
    plt.xlabel(r'$\{}$ (degrees)'.format(torname))
    plt.rc('font', **{'size':10})
    if savedir:
        plt.savefig(savedir, dpi=300)
    plt.show()


# box plots 
def box_plot(tors, tor_name, seq, savedir=None):
    # for backbones only now
    # tor_name from tor_marker
    tor_idx = np.arange(len(seq)) #[resn for resn in range(len(seq)) if sidechain_res[seq[resn]]+2>=tor_name]
    if tor_name == 1:
        tor_idx = tor_idx[1:]
    elif tor_name == 2:
        tor_idx = tor_idx[:-1]
    tor_r = np.reshape(tors, (-1, len(tor_idx))).T
    #tor_name_dict = {1:'phi', 2:'psi', 3:'chi_1', 4:'chi_2'}
    #label = tor_name_dict[tor_name]
    fig, ax= plt.subplots(figsize = (60, 10))
    ax.boxplot(list(tor_r))
    ax.set_xticklabels(np.array(tor_idx)+1)
    plt.xlabel('Residue Num')
    #plt.ylabel(r'$\{}$ (degrees)'.format(tor_name))
    if savedir is not None:
        plt.savefig(savedir, dpi=300)
    plt.show()


# correlation plots
def corr_plot(phi, psi, savedir, levels, label1='phi', label2='psi'):
    #fig, ax= plt.subplots(figsize = (8, 6))
    sns.scatterplot(phi.flatten(), psi.flatten(),  alpha=0.2, linewidth=0, s=5)
    sns.kdeplot(phi.flatten(), psi.flatten(), levels=levels, shade=True, cbar=True, 
                shade_lowest=False, cmap='cividis')
    plt.xlabel(r'$\{}$ (degrees)'.format(label1))
    plt.ylabel(r'$\{}$ (degrees)'.format(label2))
    plt.rc('font', **{'size':10})
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    #plt.rc('font', 10)
    if savedir is not None:
        plt.savefig(savedir, dpi=300)
    plt.show()

def seq_corr_plot(tor1, tor2, tor1name, seq, levels=10, savedir=None):
    if tor1name == 'phi':
        tor1 = np.reshape(tor1, (-1, len(seq)-1))[:, :-1].flatten()
        label1 = tor1name
        tor2 = np.reshape(tor2, (-1, len(seq)-1))[:, 1:].flatten()
        label2 = 'psi'
    elif tor1name == 'psi':
        # last residue does not have psi, chi1 needs to drop last depends on seq
        tor1 = np.reshape(tor1, (-1, len(seq)-1))
        tor1 = tor1[:, torsion_align('psi', 'chi1', seq)].flatten()
        label1 = tor1name
        tor2 = np.array(tor2)
        label2 = 'chi_1'
    elif tor1name == 'chi1':
        tor1 = np.reshape(tor1, (-1, np.sum(np.array([sidechain_res[res] for res in seq]) >= 1)))
        tor1 = tor1[:, torsion_align('chi1', 'chi2', seq)].flatten()
        label1 = 'chi_1'
        tor2 = np.array(tor2)
        label2 = 'chi_2'
    #print(tor1.shape, tor2.shape)
    assert tor1.shape == tor2.shape
    corr_plot(tor1, tor2, savedir, levels=levels, label1=label1, label2=label2)

# amino-related ramanchandran plots (backbone corr plot)

# pre-prolines
def pre_pro_corr_plot(data, seq, levels=10, savedir=None):
    prepro_idx = get_pre_proline_idx(seq)
    prepro_bb = get_bb_by_res(data, prepro_idx)
    corr_plot(prepro_bb['phi'], prepro_bb['psi'], savedir, levels=levels)
def non_corr_plot(data, seq, levels=10, savedir=None):
    idx = get_other_idx(seq)
    other_bb = get_bb_by_res(data, idx)
    corr_plot(other_bb['phi'], other_bb['psi'], savedir, levels=levels)
# other amino acid specific
def res_corr_plot(data, seq, res='G', levels=10, savedir=None):
    res_idx = get_res_idx(seq, res)
    res_bb = get_bb_by_res(data, res_idx)
    corr_plot(res_bb['phi'], res_bb['psi'], savedir, levels=levels)

