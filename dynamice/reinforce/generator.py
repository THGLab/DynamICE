import os
import numpy as np
import torch
import Bio.PDB

from idpconfgen.cli_build import conformer_generator
from idpconfgen.cli_build import gen_PDB_from_conformer
from idpconfgen.core.definitions import aa1to3
from dynamice.utils.utility import residue_sc_marker 


class ConformerGenerator(object):
    def __init__(self, sequence, data, psis):
        self.sequence = sequence
        self.pdb_number = 0
        self.data = data
        self.psis = psis # shape: batch, 8 of first residue
        self.res_mark = residue_sc_marker(sequence)
        self.input_seq_3_letters = [
            'HIP' if _res == 'H' else aa1to3[_res]
            for _res in self.sequence
        ]

    def _reset_global(self):
        all_omega_hidden = [0] * len(self.sequence)
        self.all_omega_hidden = all_omega_hidden
        
        all_last_ohe = [0] * (len(self.sequence)-1)
        self.all_last_ohe = all_last_ohe
        
        #self.called_residues = []
        # to record all torsions of the generated conformer
        #self.torsions = np.zeros(int(np.sum(self.res_mark)+3*(len(self.sequence)-1)))
        


    def save_pdb(self, conformer, dir_name):
        """

        Parameters
        ----------
        conformer
        dir_name: str

        Returns
        -------

        """

        self.pdb_number+=1

        pdb_string = gen_PDB_from_conformer(
            self.input_seq_3_letters,
            self.atom_labels,
            self.residue_numbers,
            np.round(conformer, decimals=3),
            )

        self.pdb_path = os.path.join(dir_name, 'gen_%i.pdb'%self.pdb_number)

        with open(self.pdb_path, 'w') as fout:
            fout.write(pdb_string)

    def parse_pdb(self, id, filename):
        """
        parse pdb file using BioPython

        Parameters
        ----------
        id
        filename

        Returns
        -------

        """
        parser = Bio.PDB.PDBParser()
        pdb_struct = parser.get_structure(id, filename)

        return pdb_struct

        
 
    def generate_torsions(self, model, temp=1, enable_torch=False):
        self._reset_global()
        
        all_angle = []
        
        for cres in range(len(self.sequence)):
            angles = self.generative_function(cres, model, temp, enable_torch)
            if cres == 0 and (not enable_torch): 
                angles = angles[2:]
            elif cres == len(self.sequence)-1 and (not enable_torch):
                # torch with grad already discards last phi
                angles = np.delete(angles, 2) 
            all_angle.append(angles)

        if enable_torch : return torch.cat(all_angle, dim=0).squeeze()
        return np.concatenate(all_angle)
    
   

    def generative_function(self, cres, model, temp, enable_torch=False):
        
        if cres == 0:
            hidden = model.init_hidden(1)
            init_tor = self.psis[np.random.randint(len(self.psis))]
            ohes = self.data.angle2tensor(init_tor.reshape(1, 8))
            ohes[ohes!=ohes] = 0. # shape: 1, 8, bins

            self.all_last_ohe[cres] = ohes 
            self.all_omega_hidden[cres + 1] = (hidden, hidden)
            init_tor[:2] = -180.0
            if enable_torch:
                return torch.tensor(np.radians(init_tor[2:self.res_mark[cres]+3]), 
                                    dtype=torch.float32, 
                                    device=ohes.device).unsqueeze(-1)
            
            return np.radians(init_tor[:self.res_mark[cres]+3]) 

        elif cres > 0 and cres < len(self.sequence)-1:
            hidden = self.all_omega_hidden[cres]
            prev_ohe = self.all_last_ohe[cres - 1]
            tor_bins, hidden = model.generate(
                prev_ohe,
                self.data.res_type[0, cres].unsqueeze(0).unsqueeze(0),
                hidden)
            ohes = []
            agls = []
            for n in range(2):
                if enable_torch:
                    tor, tor_ohe = self.data.sample_angle(tor_bins[0, n], temp)
                else:
                    tor, tor_ohe = self.data.sample_angle(tor_bins[0, n].data.cpu().numpy())
                ohes.append(tor_ohe.squeeze(0)) # shape: 1, bins
                agls.append(tor/180.*3.14159)
            in_hidden = (model.init_hidden(1, 1), model.init_hidden(1, 1))
            for n in range(2, self.res_mark[cres]+3):
                x, in_hidden = model.torsion_recurrent(ohes[-1], 
                                            self.data.tor_type[0, n-1].unsqueeze(0), 
                                            model.rnn_out, in_hidden)
                x = torch.nn.functional.softmax(x, dim=-1)
                if enable_torch:
                    tor, tor_ohe = self.data.sample_angle(x, temp)
                else:
                    tor, tor_ohe = self.data.sample_angle(x.data.cpu().numpy(),
                                        proline_filter=(self.sequence[cres]=='P' and n==3))
                # proline chis
                if n==4 and self.sequence[cres]=='P':
                    tor = proline_func(agls[-1]/3.14159*180.)
                    tor_ohe = self.data.angle2tensor(tor.reshape(1, 1))
                ohes.append(tor_ohe.squeeze(0)) # shape: 1, bins
                agls.append(tor/180.*3.14159)
            ohes = torch.stack(ohes, dim=1)
	    # ohe add paddings to non-exist sidechain
            pad_ohes = torch.zeros_like(prev_ohe, dtype=torch.float32, 
                                    device=ohes.device)[:, :5-self.res_mark[cres], :]
            self.all_last_ohe[cres] = torch.cat([ohes, pad_ohes], dim=1)
            self.all_omega_hidden[cres + 1] = hidden
            if enable_torch: 
                return torch.stack(agls)
            
            agls = np.reshape(agls, -1)
            return agls

        elif cres == len(self.sequence)-1:
            hidden = self.all_omega_hidden[cres]
            prev_ohe = self.all_last_ohe[cres - 1]
            
            tor_bins, hidden = model.generate(
                prev_ohe,
                self.data.res_type[0, -1].unsqueeze(0).unsqueeze(0),
                hidden)
            agls = []
            ohes = []
            for n in range(2):
                if enable_torch:
                    tor, tor_ohe = self.data.sample_angle(tor_bins[0, n], temp)
                else:
                    tor, tor_ohe = self.data.sample_angle(tor_bins[0, n].data.cpu().numpy())
                ohes.append(tor_ohe.squeeze(0)) # shape: 1, bins
                agls.append(tor/180.*3.14159) # shape: 1,
            in_hidden = (model.init_hidden(1, 1), model.init_hidden(1, 1))
            for n in range(self.res_mark[cres]):
                input_tor = ohes[-1]
                if n == 0: input_tor = torch.zeros_like(input_tor) 
                x, in_hidden = model.torsion_recurrent(input_tor,
                                            self.data.tor_type[0, n+2].unsqueeze(0),
                                            model.rnn_out, in_hidden)
                x = torch.nn.functional.softmax(x, dim=-1)
                if enable_torch:
                    tor, tor_ohe = self.data.sample_angle(x, temp)
                else:
                    tor, tor_ohe = self.data.sample_angle(x.data.cpu().numpy())
                ohes.append(tor_ohe.squeeze(0))
                agls.append(tor/180.*3.14159)
            if enable_torch: 
                return torch.stack(agls)
            agls = np.reshape(agls, -1)
            return np.insert(agls, 2, -np.pi)


    
    def get_conformer(self, model, energy, npdb=1, save_pdb=False, 
                      pdb_dir='gen_pdbs'):
        """
        Builds conformer through IDPConformerGenerator package.
        https://github.com/Oufan75/IDPConformerGenerator.git

        Parameters
        ----------
        generative_model
        save_pdb: bool
            Whether to generate pdb files.
        npdb: int
            Number of pdbs to generate
        energy: float
            Forcefield energy threshold to reject severe clashes
        pdb_dir: str
            The path to the directory to save pdb files.

        Returns
        -------

        """
        gen_f = lambda cres: self.generative_function(cres, model, temp=1)
            
        conf_gen = conformer_generator(input_seq=self.sequence,
                                       generative_function=gen_f,
                                       energy_threshold=energy)
        
        info = next(conf_gen)  
        self.atom_labels, self.residue_numbers, self.residue_labels = info
        
        for n in range(npdb):
            self._reset_global()
            coords = next(conf_gen)
            if save_pdb:
                if not os.path.exists(pdb_dir):
                    os.makedirs(pdb_dir)
                self.save_pdb(coords, pdb_dir)
    
   
 
 
def proline_func(x):
    # proline chi1-chi2 distributions fit from conformer pool
    coeff = [-1.32876154e-04, -0.0188184, -0.7335183,  37.1798684]
    if isinstance(x, torch.Tensor):
        x = torch.clamp(x, -60, 45)
        noise = torch.normal(0, 5, (1,), device=x.device)
    else:
        x = np.clip(x, -60, 45)
        noise = np.random.normal(0, 5)
    func = lambda x: coeff[0]*x**3+coeff[1]*x**2+coeff[2]*x+coeff[3]
    return func(x) + noise
        

