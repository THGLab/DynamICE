"""
Generates conformer ensembles with generative model
"""

from dynamice.reinforce.generator import ConformerGenerator
import numpy as np
import torch
import yaml
import os
from dynamice.data import torsion_loader#, split_data
from dynamice.models import RecurrentModel
from dynamice.utils import generate_tor_marker


import logging
logger = logging.getLogger('numba')
logger.setLevel(logging.WARNING)


drk_seq  = 'MEAIAKHDFSATADDELSFRKTQILKILNMEDDSNWYRAELDGKEGLIPSNYIEMKNHD'
asyn_seq = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAV\
VTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'     

IDP_SEQ = drk_seq
settings_path = 'local/training_25/run_scripts'
settings = yaml.safe_load(open(settings_path+'/torsion_recurrent.yml', "r"))

device = torch.device('cuda:0')
# data
#root = settings['data']['root'].strip()
#data = np.load(os.path.join(root, 'bbsc.npy'))
#_, _, test = split_data(data[:7373], train_size=5000, val_size=500, seed=1, 
#                  save_path=settings_path)

test = np.load(os.path.join(settings_path, 'test_bbsc.npy'))
smearing = (settings['data']['start_val'],
            settings['data']['stop_val'],
            settings['data']['ohe_size'])
test_gen = torsion_loader(test,
                         sequence=IDP_SEQ,
                         embedding=3,
                         smearing=smearing,
                         gaussian=settings['data']['gaussian'],
                         gaussian_margin=settings['data']['gaussian_margin'],
                         gaussian_normalize=settings['data']['gaussian_normalize'],
                         gaussian_factor=settings['data']['gaussian_factor'],
                         batch_size=settings['training']['val_batch_size'],
                         device=device,
                         shuffle=True) #settings['training']['shuffle'])
test_batch, _ = next(test_gen)

# model
model = RecurrentModel(recurrent=settings['model']['recurrent'],
                          filter_in=settings['data']['ohe_size'],
                          n_filter_layers=settings['filter']['n_layer'],
                          filter_out=settings['filter']['out'],
                          filter_drop=settings['filter']['dropout'],
                          rec_stack_size=settings['model']['rec_stack_size'],
                          rec_neurons_num=settings['model']['rec_neurons_num'],
                          rec_dropout=settings['model']['rec_dropout'],
                          embed_out=settings['filter']['embedding'],
                          embed_in=3)
            
output_path = settings['general']['output']
out_path = os.path.join(output_path[0], output_path[1], 'training_%i'%output_path[2], 'reinforce')
model.load_state_dict(torch.load(os.path.join(out_path, 'best_model.tar')))
model.to(device)
model.eval()

generator = ConformerGenerator(IDP_SEQ, test_batch, test[:, :8])
#generator.get_conformer(model, npdb=20, energy=250, save_pdb=True,
#                        pdb_dir=os.path.join(out_path, 'pdbs'))


phi = []
psi = []
omg = []
chi1 = []
chi2 = []
tor_mark = generate_tor_marker(IDP_SEQ) 
for n in range(100):
    agls = np.degrees(generator.generate_torsions(model))
    phi += list(agls[np.argwhere(tor_mark == 1).flatten()])
    psi += list(agls[np.argwhere(tor_mark == 2).flatten()])
    omg += list(agls[np.argwhere(tor_mark == 0).flatten()])
    chi1 += list(agls[np.argwhere(tor_mark == 3).flatten()])
    chi2 += list(agls[np.argwhere(tor_mark == 4).flatten()])


# torsion histograms
#from dynamice.utils.plotting import torsion_hist
#torsion_hist(phi, 'phi')#, os.path.join(out_path, 'phi.png'))
#torsion_hist(psi, 'psi')#, os.path.join(out_path, 'psi.png'))
#torsion_hist(omg, 'omega')#, os.path.join(out_path, 'omega.png'))
#torsion_hist(chi1, 'chi1')#, os.path.join(out_path, 'chi1.png'))
#torsion_hist(chi2, 'chi2')#, os.path.join(out_path, 'chi2.png'))





