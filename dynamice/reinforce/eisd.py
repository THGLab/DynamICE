import os
import numpy as np
import pandas as pd
#import shutil

from eisd.scorers import (jc_optimization_ensemble, 
                          noe_optimization_ensemble,
                          pre_optimization_ensemble,
                          fret_optimization_ensemble,
                          cs_optimization_ensemble)


class EISD_API(object):
    """
    This is the API to the EISD scoring to calculate maximum log-likelihood of a
    disordered protein ensemble.
    Parameters
    ----------
    exp_data
    bc_data
    property: list
        List of properties that you want to back calculate.
            value: any of ('jc', 'noe', 'pre', 'fret', 'rdc', 'rh')
    """
    def __init__(self, exp_data, bc_data, property):

        self.exp_data = exp_data
        self.bc_data = bc_data
        self.property = property


    def init_ensemble(self):
        for prop in self.property:
            if prop == 'jc':
                self.jc = JC(self.exp_data, self.bc_data)

            elif prop == 'noe':
                self.noe = NOE(self.exp_data, self.bc_data)

            elif prop == 'pre':
                self.pre = PRE(self.exp_data, self.bc_data)

            elif prop == 'fret':
                self.fret = FRET(self.exp_data, self.bc_data)   
            
            elif prop == 'cs':
                self.cs = CS(self.exp_data, self.bc_data)
      
    


class JC(object):
    def __init__(self, exp_data, bc_data):
        self.exp_data = exp_data
        self.bc_data = bc_data

    def eisd(self, ens_bc):

        assert ens_bc.ndim == 2

        # update bc_data
        self.bc_data['jc'].data = pd.DataFrame(ens_bc)

        # eisd
        indices = np.arange(ens_bc.shape[0])
        serror, eisd_score = jc_optimization_ensemble(self.exp_data, self.bc_data, indices)[:2]
        return eisd_score, serror


    def back_calculator(self, torsions_phi, exp_indices=None):
        # for testing noe dist restrain only
        assert torsions_phi.ndim == 2
        alpha = np.cos(torsions_phi - np.radians(60))

        if exp_indices is None:
            return alpha
        else:
            return alpha[:, exp_indices]

    def ensemble(self, ens_phi):
        """
        Parameters
        ----------
        ens_phi: np.ndarray
            2D array of phi torsion angles for the conformers in the emsemble.
        Returns
        -------
        np.ndarray: back calculated data for a given ensemble
        np.ndarray: the indices of back calculated data that their experimental value is available.
        """

        # back calculate jc for a given ensemble of backbone phi torsion angles
        exp_indices = (self.exp_data['jc'].data.resnum.values-2).astype(np.int)
        ens_bc = self.back_calculator(ens_phi, exp_indices)

        return ens_bc 

  


class NOE(object):
    def __init__(self, exp_data, bc_data, back_bone=False):

        self.exp_data = exp_data
        self.bc_data = bc_data
   


    def eisd(self, ens_bc):

        # update bc_data
        self.bc_data['noe'].data = pd.DataFrame(ens_bc)

        # eisd
        indices = np.arange(ens_bc.shape[0])
        se, eisd_score, avg_dis, eisd_per_data = noe_optimization_ensemble(self.exp_data, self.bc_data, indices)

        return eisd_score, se

    def back_calculator(self, structure, chain="A"):
        """
        Parameters
        ----------
        structure: Bio.PDB.Structure.Structure
        chain: str
        Returns
        -------
        np.ndarray: 2D array of back calculated distances corresponding to experimental data.
        """

        exp_noe = self.exp_data['noe'].data
        distances = []
        for idx in range(exp_noe.shape[0]):
            row = exp_noe.iloc[idx]
            avg_distance = self.get_average_distance(int(row.res1),
                                                     row.atom1,
                                                     row.atom1_multiple_assignments,
                                                     int(row.res2),
                                                     row.atom2,
                                                     row.atom2_multiple_assignments,
                                                     chain,
                                                     structure)
            distances.append(avg_distance)

        return distances

    def get_average_distance(self, res1, atom1_name, atom1_multiple_assignments,
                             res2, atom2_name, atom2_multiple_assignments,
                             chain, structure):

        atom1_list = []
        for atom in structure[0][chain][res1]:
            if atom1_name == 'H':
                atom1_list.append(structure[0][chain][res1]['H'])
                break
            if atom1_name in atom.get_name():
                atom1_list.append(atom)
            if not atom1_multiple_assignments and len(atom1_list)==1:
                break
        #print(atom1_name, atom1_list)
        atom2_list = []
        for atom in structure[0][chain][res2]:
            if atom2_name == 'H':
                atom2_list.append(structure[0][chain][res2]['H'])
                break
            if atom2_name in atom.get_name():
                atom2_list.append(atom)
            if not atom2_multiple_assignments and len(atom2_list)==1:
                break
        #print(atom2_name, atom2_list)
        combos = 0.0
        num_combos = 0
        for first_atom in atom1_list:
            for second_atom in atom2_list:
                combos = combos + (first_atom - second_atom)**(-6.)
                num_combos = num_combos + 1

        d = (combos/float(num_combos))**(-1/6)

        return d

    def ensemble(self, ens_structures, chain = "A"):
        """
        Parameters
        ----------
        ens_structures: list
            list of structure in Bio.PDB.Structure.Structure
        chain: str
            the chain name in the pdb file
        Returns
        -------
        np.ndarray: back calculate data for a given ensemble
        """
        ens_bc = []
        for structure in ens_structures:
            ens_bc.append(self.back_calculator(structure, chain))

        return np.array(ens_bc)


class PRE(object):
    def __init__(self, exp_data, bc_data, back_bone=False):

        self.exp_data = exp_data
        self.bc_data = bc_data

    def eisd(self, ens_bc):

        # update bc_data
        self.bc_data['pre'].data = pd.DataFrame(ens_bc)

        # eisd
        indices = np.arange(ens_bc.shape[0])
        se, eisd_score, _, eisd_per_data = pre_optimization_ensemble(self.exp_data, self.bc_data, indices)

        return eisd_score, se



    def back_calculator(self, structure, chain="A"):
        """
        Parameters
        ----------
        structure: Bio.PDB.Structure.Structure
        chain: str
        Returns
        -------
        np.ndarray: 2D array of back calculated distances corresponding to experimental data.
        """

        exp_pre = self.exp_data['pre'].data
        distances = []
        for idx in range(exp_pre.shape[0]):
            row = exp_pre.iloc[idx]
            avg_distance = self.get_average_distance(int(row.res1),
                                                     row.atom1,
                                                     int(row.res2),
                                                     row.atom2,
                                                     chain,
                                                     structure)
            distances.append(avg_distance)
        #print('actual dis', distances)
        distances = np.array(distances).reshape(1,-1)

        return distances

    def get_average_distance(self, res1, atom1_name, res2, atom2_name,
                             chain, structure):

        atom1 = structure[0][chain][res1][atom1_name]
        atom2 = structure[0][chain][res2][atom2_name]
 
        return atom1 - atom2

    def ensemble(self, ens_structures, chain="A"):
        """
        this function is hard coded for the drksh3 data.
        Parameters
        ----------
        ens_structures: list
            list of structure in Bio.PDB.Structure.Structure
        chain: str
            the chain name in the pdb file
        Returns
        -------
        np.ndarray: back calculate data for a given ensemble
        """
        ens_bc = []
        for structure in ens_structures:
            ens_bc.append(self.back_calculator(structure, chain))
        ens_bc = np.concatenate(ens_bc, axis=0)

        return ens_bc


class FRET(object):
    def __init__(self, exp_data, bc_data):

        self.exp_data = exp_data
        self.bc_data = bc_data

    def eisd(self, ens_bc):

        # update bc_data
        self.bc_data['fret'].data = pd.DataFrame(ens_bc)

        # eisd
        indices = np.arange(ens_bc.shape[0])
        se, eisd_score, _, eisd_per_data = fret_optimization_ensemble(self.exp_data, self.bc_data, indices)

        return eisd_score, se

    def back_calculator(self, structure, chain="A"):
        """
        Parameters
        ----------
        structure: Bio.PDB.Structure.Structure
        chain: str
        Returns
        -------
        np.ndarray: 2D array of back calculated distances corresponding to experimental data.
        """

        res1s = self.exp_data['fret'].data.res1.values
        res2s = self.exp_data['fret'].data.res2.values
        efficiency = []
        for n in range(self.exp_data['fret'].data.shape[0]):
            res1 = np.int(res1s[n])
            res2 = np.int(res2s[n])
            scale = self.exp_data['fret'].data.scale.values[n]
            dist = structure[0][chain][res1]['CA'] - structure[0][chain][res2]['CA']
            efficiency.append(1.0/(1.0+(dist/scale)**6.0))

        return efficiency


    def ensemble(self, ens_structures, chain="A"):
        """
        this function is hard coded for the drksh3 data.
        Parameters
        ----------
        ens_structures: list
            list of structure in Bio.PDB.Structure.Structure
        chain: str
            the chain name in the pdb file
        Returns
        -------
        np.ndarray: 2D array of back calculate data for a given ensemble with shape (ens_size, 1)
        """
        ens_bc = []
        for structure in ens_structures:
            ens_bc.append(self.back_calculator(structure, chain))

        ens_bc = np.array(ens_bc)

        return ens_bc




class CS(object):
    def __init__(self, exp_data, bc_data, cspred_path='/home/oufan/Desktop/CSpred-master/CSpred.py'):
        self.exp_data = exp_data
        self.bc_data = bc_data
        self.cspred_path='/home/oufan/Desktop/CSpred-master/CSpred.py'
        self.atommap = {'H': 2, 'HA': 3, 'C': 4, 'CA': 5, 'CB': 6, 'N': 7}


    def eisd(self, ens_bc):

        # update bc_data
        self.bc_data['cs'].data = pd.DataFrame(ens_bc)

        # eisd
        indices = np.arange(ens_bc.shape[0])
        se, eisd_score = cs_optimization_ensemble(self.exp_data, self.bc_data, indices)[:2]
        return eisd_score, se
    
    
    def back_calculator(self, pdb_pointer):
        # create temporary folder
        tmp_folder = os.path.dirname(pdb_pointer) #tempfile.mkdtemp()

        # run UCBShift command
        tmp_out = os.path.join(tmp_folder, 'cs.csv')
        #tmp_pdb = os.path.join(tmp_folder, 'in.pdb')
        #os.system("cp %s %s" % (pdb_pointer, tmp_pdb))
        # replace HIP with HIS
        os.system("sed -i 's/HIP/HIS/g\' %s" % pdb_pointer)     
        os.system("python %s %s -o %s -x" % (self.cspred_path, pdb_pointer, tmp_out))

        # parse output
        shifts = []
        output = pd.read_csv(tmp_out)
        exp_cs = self.exp_data['cs'].data
        for idx in range(exp_cs.shape[0]):
            resnum = int(exp_cs.iloc[idx].resnum)
            atom = self.atommap[exp_cs.iloc[idx].atomname]
            shifts.append(output[output.RESNUM==resnum].values[0, atom])
            
        # clean after
        #shutil.rmtree(tmp_folder)
        return np.reshape(shifts, (1, -1))
    
    
    def ensemble(self, ens_structures):
        """
        this function is hard coded for the drksh3 data.
        Parameters
        ----------
        ens_structures: list
            list of pdb locations
        Returns
        -------
        np.ndarray: back calculate data for a given ensemble
        np.ndarray: residue numbers (indices, 1-indexed) of experimental data
        """
   
        ens_bc = []
        # inputs are backcalculations (shape: ens_size, seq_len, 6)        
        if isinstance(ens_structures, np.ndarray):
            exp = self.exp_data['cs'].data
            for n in range(exp.values.shape[0]):
                resnum = exp.iloc[n].resnum
                atom = self.atommap[exp.iloc[n].atomname] - 2
                ens_bc.append(ens_structures[:, resnum-1, atom].reshape(1, -1))
            return np.concatenate(ens_bc, axis=0).T
        
        # inputs are pdb locations
        for structure in ens_structures:
            ens_bc.append(self.back_calculator(structure))
        ens_bc = np.concatenate(ens_bc, axis=0)

        return ens_bc


 
