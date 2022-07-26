import numpy as np
import torch
#import torch.nn.functional as F
#from torch.distributions import Categorical
from dynamice.utils.utility import (
  #get_batched_angles,
  get_batched_coords,
  get_scn_angles,
  get_scn_structure,
  residue_sc_marker)
from dynamice.reinforce.backcalc import (
  find_coordpair_idx,
  calc_distances,
  calc_jc, 
  calc_smfret,
  )

  

mseloss = torch.nn.MSELoss(reduction='mean')

# 'Floating' loss to account for the uncertainties in exp_data
def FBloss(bc, exp, errors):
    # process errors
    if len(errors) != 2:
        errors = (errors/2., errors/2.)
    
    # adjust exp_data if bc_data within uncertainties
    bc_copy = bc.clone().detach().cpu().numpy()
    exp = exp.copy()
    low_mask = np.less(bc_copy, exp-errors[0]) 
    up_mask = np.greater(bc_copy, exp+errors[1])
    
    # adjust exp to edge values
    exp[up_mask] += errors[1][up_mask]
    exp[low_mask] -= errors[0][low_mask] 
    tmask = np.logical_or(low_mask, up_mask)
    c = tmask.sum()/bc_copy.shape[-1] #normalize
    
    return c*mseloss(bc[tmask], torch.tensor(exp[tmask], dtype=torch.float32, device=bc.device))


class Reinforcement(object):
    def __init__(self, sequence, generator, conformer_generator, 
                 eisd, optimizer, device):
        """
        Constructor for the Reinforcement object.
        Parameters
        ----------
        generator: object of type RecurrentModel
            generative model that produces torsion angles (trajectories)
        conformer_generator: object of type ConformerGenerator
            conformer_generator gets the generative model and generate a conformer and checks
            the underlying rules.
        eisd: object of type dynamice.reinforce.EISD
            eisd accepts a conformer and returns a numerical
            score the can be used as a reward.
        Returns
        -------
        object of type Reinforcement used for biasing the properties estimated
        by the predictor of trajectories produced by the generator to maximize
        the custom reward function get_reward.
        """

        super(Reinforcement, self).__init__()
        self.generator = generator
        self.conformer_generator = conformer_generator
        self.eisd = eisd
        self.sequence = sequence
        self.optimizer = optimizer
        self.device = device
        self.init_coeff = {}
        for props in self.eisd.property:
            self.init_coeff[props] = None

        global res_mark
        res_mark = residue_sc_marker(sequence)
        
        
        

    def get_exp_data(self, atom_labels):
        exp_data = {'error': {}}
        idxs = {}
        for prop in self.eisd.property:
            # initialize all exp data for loss function and validations
            if prop in ['pre', 'noe']:
                exp_data[prop] = self.eisd.exp_data[prop].data['dist_value'].values    
                exp_data['error'][prop] = (self.eisd.exp_data[prop].data['lower'].values, #0.) 
                                           self.eisd.exp_data[prop].data['upper'].values)
                idxs[prop] = find_coordpair_idx(self.eisd.exp_data[prop].data, atom_labels)
            elif prop == 'fret':
                exp_data[prop] = self.eisd.exp_data[prop].data['value'].values
                exp_data['error'][prop] = self.eisd.exp_data[prop].data['error'].values
                idxs[prop] = find_coordpair_idx(self.eisd.exp_data[prop].data, atom_labels)
            elif prop == 'jc':        
                exp_data[prop] = self.eisd.exp_data['jc'].data['value'].values
                #exp_data[prop] = torch.tensor(exp_jc, dtype=torch.float32, device=self.device)
                exp_data['error'][prop] = self.eisd.exp_data['jc'].data['error'].values
                idxs[prop] = self.eisd.exp_data['jc'].data['resnum'].values.astype(int) - 2
        return exp_data, idxs
    
 
 
    def train(self, batch_size, exp_data, idxs, coeff,
                          temp=1, ens_size=100, grad_clipping=None):
        torsions = []
        bc_list = {}
        for prop in self.eisd.property:
            bc_list[prop] = []
   
        self.generator.train()

        for _ in range(batch_size):
            # Sampling new trajectory
            angles = self.conformer_generator.generate_torsions(self.generator, temp, True)         
            torsions.append(angles)
        torsions_rad = torch.stack(torsions, dim=0)
        # use sidechainnet to convert to cartesian coordinates
        coords, batch_agls = get_batched_coords(torsions_rad, self.sequence, res_mark, 
                                    self.device, hydrogen=True)
        coords = coords.reshape(batch_size, len(res_mark), 24, 3) 
        
        for n in range(batch_size):
            for prop in coeff:
                # initialize the opt data types
                if prop in ['noe', 'pre']:
                    avg_distance = calc_distances(coords[n], idxs[prop])
                    bc_list[prop].append(avg_distance)
                elif 'fret' == prop:
                    dist = calc_distances(coords[n], idxs[prop])
                    eff = calc_smfret(#self.eisd.exp_data[prop].data.res1.values,
                          dist, 
                          self.eisd.exp_data[prop].data.scale.values)
                    bc_list[prop].append(eff)
                elif 'jc' == prop:
                    jcoups = calc_jc(batch_agls[n, 1:, 0], idxs[prop])
                    bc_list['jc'].append(jcoups)
              
        # calculate MSE loss between back-calculations and exp_bc
        loss = 0.
        for prop in coeff:
            if prop in ['noe', 'pre']:
                dist_list = torch.mean(torch.stack(bc_list[prop], dim=0)**(-6), dim=0)**(-1./6.)
                dloss = FBloss(dist_list, exp_data[prop], exp_data['error'][prop]) 
                        #mseloss(dist_list, torch.tensor(exp_data[prop], dtype=torch.float32, device=self.device))
                        #FBloss(dist_list, exp_data[prop], exp_data['error'][prop]) 
                if self.init_coeff[prop] is None:
                    self.init_coeff[prop] = 0.2*dloss.detach()
                loss += coeff[prop]*dloss/self.init_coeff[prop]
            elif prop in ['jc', 'fret']:
                jc_list = torch.mean(torch.stack(bc_list[prop], dim=0), dim=0)
                jloss = FBloss(jc_list, exp_data[prop], exp_data['error'][prop]) 
                        #mseloss(jc_list, torch.tensor(exp_data[prop], dtype=torch.float32, device=self.device))
                        #FBloss(jc_list, exp_data[prop], exp_data['error'][prop]) 
                if self.init_coeff[prop] is None:
                    self.init_coeff[prop] = 0.2*jloss.detach()
                loss += coeff[prop]*jloss/self.init_coeff[prop]


        self.optimizer.zero_grad()
        # Doing backward pass and parameters update
        loss.backward(retain_graph=True)
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                                                   grad_clipping)
        self.optimizer.step()
        
        #validation
        self.generator.eval()
        coords = []
        phis = []
        for _ in range(ens_size):
            # Sampling new trajectory
            angles = self.conformer_generator.generate_torsions(self.generator, False)
            torsions = get_scn_angles(angles, res_mark, self.sequence)
            phis.append(torsions[1:, 0])
            structure = get_scn_structure(torsions, self.sequence, True)
            coords.append(structure)
        coords = np.reshape(coords, (ens_size, len(res_mark), 24, 3))
        jc_ens = self.eisd.jc.ensemble(np.array(phis))

        # val loss as EISD score
        val_loss = {}
        if 'noe' in self.eisd.property:
            ens_bc = []
            for n in range(ens_size):
                ens_bc.append(calc_distances(coords[n], idxs['noe']))
            val_loss['noe']=self.eisd.noe.eisd(np.array(ens_bc))[0]
        if 'pre' in self.eisd.property:
            ens_bc = []
            for n in range(ens_size):
                ens_bc.append(calc_distances(coords[n], idxs['pre']))
            val_loss['pre']=self.eisd.pre.eisd(np.array(ens_bc))[0]
        if 'fret' in self.eisd.property:
            ens_bc = []
            for n in range(ens_size):
                effs = calc_smfret(calc_distances(coords[n], idxs['fret']), 
                          self.eisd.exp_data['fret'].data.scale.values)
                ens_bc.append(effs)
            val_loss['fret']=self.eisd.fret.eisd(np.array(ens_bc))[0]
        val_loss['jc']=self.eisd.jc.eisd(jc_ens)[0]

        return loss.item(), val_loss
        
 
    
  

