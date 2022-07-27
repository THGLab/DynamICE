import os
import numpy as np
import yaml
import torch
from torch.optim import Adam
from eisd.parser import read_data
from eisd.utils import meta_data

from dynamice.data import torsion_loader
from dynamice.models import RecurrentModel
from dynamice.reinforce import Reinforcement, EISD_API
from dynamice.reinforce.generator import ConformerGenerator
from dynamice.utils.utility import get_scn_labels

import logging
logger = logging.getLogger('numba')
logger.setLevel(logging.WARNING)


drkseq = 'MEAIAKHDFSATADDELSFRKTQILKILNMEDDSNWYRAELDGKEGLIPSNYIEMKNHD'
asynseq = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAV\
VTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'
IDP_SEQ = drkseq

settings_path = 'local/training_1/run_scripts'
settings = yaml.safe_load(open(settings_path+'/torsion_recurrent.yml', "r"))
device = torch.device('cuda:0')
test = np.load(os.path.join(settings_path, 'test_bbsc.npy'))
smearing = (settings['data']['start_val'],
            settings['data']['stop_val'],
            settings['data']['ohe_size'])

test_gen = torsion_loader(test,
                         IDP_SEQ,
                         embedding=3,
                         smearing=smearing,
                         gaussian=settings['data']['gaussian'],
                         gaussian_margin=settings['data']['gaussian_margin'],
                         gaussian_normalize=settings['data']['gaussian_normalize'],
                         gaussian_factor=settings['data']['gaussian_factor'],
                         batch_size=settings['training']['val_batch_size'],
                         device=device,
                         shuffle=False) #settings['training']['shuffle'])

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
out_path = os.path.join(output_path[0], 'training_%i'%output_path[1])
model.load_state_dict(torch.load(os.path.join(out_path, 'best_model.tar')))
model.to(device)


# instantiate conformer generator
data, _ = next(test_gen)
conformer_generator = ConformerGenerator(IDP_SEQ, data, test[:, :8])


# properties meta data and back calculation
metadata_path = '/home/oufan/Desktop/X-EISD/data'
filenames = meta_data(metadata_path)
exp_data = read_data(filenames['exp'], mode='exp')
bc_data = read_data(filenames['mcsce'], mode='mcsce')

# initial ensemble
props = ['jc', 'noe'] #'fret','pre', 'cs',
eisd = EISD_API(exp_data, bc_data, props)
eisd.init_ensemble()
del exp_data, bc_data

# instantiate reinforcement
optimizer = Adam([{'params':model.linear_phi.parameters(), 'lr': 0.0005}, 
                  {'params':model.linear_psi.parameters(), 'lr': 0.0005},
                  {'params':model.linear_out.parameters(), 'lr': 0.},
                  {'params':model.tor_filter.parameters(), 'lr': 0.},
                  {'params':model.tor_recurrent.parameters(), 'lr':0.},
                  {'params':model.res_embedding.parameters(), 'lr':0},
                  {'params':model.filter.parameters(), 'lr':0},
                  {'params':model.res_recurrent.parameters(), 'lr':0.},
                  {'params':model.linear_omega.parameters(), 'lr':0.}],
                 amsgrad=True)

#optimizer = Adam(model.parameters(),
#                 lr=0.0005,
#                 weight_decay=0.01)

output_dir = os.path.join(out_path, 'reinforce')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
RL = Reinforcement(sequence=IDP_SEQ,
                  generator=model,
                  conformer_generator=conformer_generator,
                  eisd=eisd,
                  optimizer=optimizer,
                  device=device)


# RL training
  
def rank_reward(reward_list, ratio):
    #if hard < 30: return -1e4
    '''
    Calculates total current score for specified properties with ratios
    Parameters
    ==============
    ratio: dict
        {key[property]: [offset, r]}
        score = (reward - offset)/r
    '''
    score = 0.
    for key in ratio:
        if reward_list[key][-1] < ratio[key][0]: return -1e4
        score += (reward_list[key][-1] - ratio[key][0])/ratio[key][1]
    return score


n_iterations = 200
rewards = {'jc': [], 'noe': [], 'pre': []}
# optimization weight hyperparameters
coeffs = {'noe': 4, 'jc': 1} 
temp_schedule = [(0.985**n) for n in range(n_iterations)]
last_score = 0
saved_idx = -1
# read exp data
scnlabels = get_scn_labels(IDP_SEQ)
exp, exp_idxs = RL.get_exp_data(scnlabels)


for i in range(n_iterations):
    # Reinforce algorithm
    loss, eisd = RL.train(batch_size=50, exp_data=exp,
                                idxs=exp_idxs, 
                                temp = temp_schedule[i],
                                grad_clipping = 2.,
                                coeff=coeffs, ens_size=100)
    for key in coeffs:
        rewards[key].append(eisd[key])
    
    print('iter:', i+1,
          'loss:', np.round(loss, 2),
          'temp:', np.round(temp_schedule[i], 2),
          'jc:', np.round(eisd['jc'], 2),
          'noe:', np.round(eisd['noe'], 2),
          #'pre:', np.round(eisd['pre'], 2),
          )   

    curr_score = rank_reward(rewards, ratio={'jc': [-120, 1], 'noe': [450, 5]})
    if curr_score > last_score or i == 0:
        saved_idx = i
        last_score = curr_score
        torch.save(RL.generator.state_dict(), os.path.join(output_dir, 'model_noejc.tar'))
        
print('saved checkpoint:', saved_idx+1)
print('done')
