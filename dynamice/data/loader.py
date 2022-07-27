import numpy as np
import torch
from dynamice.utils import (onehot_digitizer, GaussianSmearing)
from dynamice.utils.utility import ( 
# residue_sc_marker,
# generate_tor_marker, 
 get_unrepeated_res,
 get_res_idx)
torch.manual_seed(8)



def res_type_generate(seq, setseq=False):
    '''
    generates residue types for recurrent model.
    
    Parameter:
        setseq [bool]: whether to set order of amino code
    Return:
        one-hot encoder for amino acid sequence (shape: sequence length, 20)
        
    '''
    seq_pos = ''.join(get_unrepeated_res(seq)) 
    if setseq: seq_pos = 'AGILPFWYDERHKSTMNQVC'  
    res_type = np.zeros((len(seq), len(seq_pos)))
    for res in get_unrepeated_res(seq) :
        res_idx = seq_pos.index(res)
        res_type[get_res_idx(seq, res), res_idx] = 1.0

        
    return res_type

    
def sample_from_dist(x, n_bins, iftorch, temp):
    # sample from distribution x and return bin index
    # torch enable grad uses gumber softmax
    if iftorch:
        logp = torch.log(x)
        ohe = torch.nn.functional.gumbel_softmax(logp, temp, True)
        idx = torch.sum(torch.arange(n_bins, device=x.device)*ohe)
    else:
        idx = np.random.choice(n_bins, 1, p=x)[0]
    return idx


class BatchDataset(object):
    """
    Parameters
    ----------
    input: dict
        The dictionary of batch data in ndarray format.

    device: torch.device
        the device can be cpu or cuda based on the availability

    """
    def __init__(self, input, device):
        """

        Parameters
        ----------
        input: dict
        device: torch.device
        """
        self.meta = input['meta']
        self.device = device

        self.structures = torch.tensor(input['structures'], dtype=torch.float32, device=device)

        if 'tor_idx' in input:
            self.tor_type = torch.tensor(input['tor_idx'], dtype=torch.long, device=device) 
        elif 'tor_type' in input:
            self.tor_type = torch.tensor(input['tor_type'], dtype=torch.float32, device=device) 

        if 'res_idx' in input:
            self.res_type = torch.tensor(input['res_idx'], dtype=torch.long, device=device) 
        elif 'res_type' in input:
            self.res_type = torch.tensor(input['res_type'], dtype=torch.float32, device=device) 


    def __getitem__(self, index):

        output = dict()
        output['structures'] = self.structures[index]

        return output

    def __len__(self):
        return self.structures.size()[0]

    def angle2tensor(self, angle):
        """

        Parameters
        ----------
        angle: ndarray
            numpy array with shape (n_batch, n_angles)

        Returns
        -------
        ndarray: 3D numpy array with shape (n_batch, n_angles, n_bins)

        """
        smearing = self.meta['smearing']
        gaussian = self.meta['gaussian']
        gaussian_margin = self.meta['gaussian_margin']
        gaussian_normalize = self.meta['gaussian_normalize']
        gaussian_factor = self.meta['gaussian_factor']

        if gaussian:
            gs = GaussianSmearing(smearing[0], smearing[1], smearing[2],
                                  margin=gaussian_margin,
                                  normalize=gaussian_normalize,
                                  width_factor=gaussian_factor)
            data = gs.forward(angle)  # n_data, n_torsions, n_gaussians
            assert data.ndim == 3
            assert data.shape[2] == smearing[2]

        else:
            # torch with grad not implemented
            # onehot encoder
            data = onehot_digitizer(angle, smearing, binary=True)
            assert data.ndim == 3
            assert data.shape[2] == smearing[2]

        return torch.tensor(data, dtype=torch.float32, device=self.device) 

    def angle2ohe(self, angle):
        if not np.any(angle > np.pi):
            angle = np.degrees(angle)
        smearing = self.meta['smearing']
        data = onehot_digitizer(angle, smearing, binary=True)
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def sample_angle(self, x, temp=1,
                     omega_filter=False, proline_filter=False):
        
        n_bins = x.shape[-1]
        bin_size = 360 / n_bins
        std = self.meta['gaussian_factor']
        x = x.reshape(-1)
        iftorch = isinstance(x, torch.Tensor)
        normal = torch.normal if iftorch else np.random.normal
        idx = sample_from_dist(x, n_bins, iftorch, temp) 
        noise = normal(bin_size/2., std, (1,), device=x.device) if iftorch else normal(bin_size/2., std, (1,)) 
        angle = idx * bin_size + noise - 180. #shape (1,)
      
        # deals with angles with special ranges
        if omega_filter and not iftorch:
            count = 0
            while abs(angle) < 155:
                idx = sample_from_dist(x, n_bins, iftorch, temp)
                angle = idx * bin_size + normal(bin_size/2., std, (1,)) - 180.
                count += 1
                if count > 5:
                    #print('omega out of bound, randomly selected')
                    angle = normal(360, 6, (1,))  
                    if angle > 360: 
                        angle -= 540 
                    else:
                        angle -= 180
                    break
                
        if proline_filter and not iftorch:
            count = 0
            while abs(angle) > 50:
                idx = sample_from_dist(x, n_bins, iftorch, temp)
                angle = idx * bin_size + normal(bin_size/2., std, (1,)) - 180.
                count += 1
                if count > 5:
                    #print('proline chi1 out of bound, randomly selected')
                    if np.random.random() < 0.5:
                        angle = normal(30., 3., (1,)) 
                    else:
                        angle = normal(-25., 5., (1,)) 
                    break
                

        angle_vector = self.angle2tensor(angle.reshape(1, 1)) # 1,1,n_bins
        return angle, angle_vector
    


def torsion_loader(data,
                   sequence,
                   smearing,
                   embedding=False,
                   gaussian=False,
                   gaussian_margin=0,
                   gaussian_normalize=False,
                   gaussian_factor=1.5,
                   batch_size=32,
                   device=None,
                   shuffle=False,
                   drop_last=False):
    r"""
    The main function to load and iterate protein backbone torsion angles.

    Parameters
    ----------
    data: np.ndarray
        data array with shape(N, n_atoms)

    smearing: tuple
        (start, stop, n_bins)

    gaussian: bool
        If False smearing returns one-hot-encodings,
        if True, the gaussian smearing will be returned.

    gaussian_margin: int, optional (default: 0)
        The margin for symmetric gaussian smearing (for angles)

    batch_size: int, optional (default: 32)
        The size of output tensors

    device: torch.device
        either cpu or gpu (cuda) device.

    shuffle: bool, optional (default: True)
        If ``True``, shuffle the list of file path and batch indices between iterations.

    drop_last: bool, optional (default: False)
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)

    Yields
    -------
    BatchDataset: instance of BatchDataset with the all batch data

    """
    
    
    if gaussian:
        gs = GaussianSmearing(smearing[0], smearing[1], smearing[2],
                              margin=gaussian_margin,
                              normalize=gaussian_normalize,
                              width_factor=gaussian_factor)
        data = gs.forward(data)  # n_data, n_torsions, n_gaussians

    else:
        # onehot encoder
        data = onehot_digitizer(data, smearing, binary=True)
        assert data.ndim == 3
        assert data.shape[2] == smearing[2]

    
    # Datasets padded with Nan for non-applicable torsion angles
    # replace Nan with zero
    data = np.nan_to_num(data)

    n_data = data.shape[0]
    #n_sequence = len(sequence)  
    n_steps = int(np.ceil(n_data/batch_size))
    
    if shuffle:
        shuffle_idx = np.arange(n_data)
        np.random.shuffle(shuffle_idx)
        data = data[shuffle_idx]
    
    if embedding == 3:
        # embed pre, current, post amino acid 20**3
        baser = res_type_generate('A'+sequence+'A', True).argmax(-1)
        res_type = []
        for i in range(len(sequence)):
            idx = baser[i] + baser[i+1]*20 + baser[i+2]*(20**2)
            res_type.append(idx)
    elif embedding == 2:
        # embed current, post amino acid 20**2
        baser = res_type_generate(sequence+'A', True).argmax(-1)
        res_type = []
        for i in range(len(sequence)):
            idx = baser[i] + baser[i+1]*20 
            res_type.append(idx)
    elif embedding == 1:
        res_type = res_type_generate(sequence)
        res_type = res_type.argmax(-1)
    else:
        res_type = res_type_generate(sequence)  

    # torsion type, shape (8, 8)
    tor_type = np.eye(8)    # omega, phi, psi, chi1 - chi5




    # iterate over data
    while True:

        complete_epoch = False
        split = 0

        while (split + 1) * batch_size <= n_data:
            # Output a batch
            data_batch = data[split * batch_size: (split + 1) * batch_size]

            batch_dataset = {
                'structures': data_batch,
                'meta': {'smearing': smearing,
                         'gaussian': gaussian,
                         'gaussian_margin':gaussian_margin,
                         'gaussian_normalize': gaussian_normalize,
                         'gaussian_factor': gaussian_factor},
                'tor_type': np.tile(tor_type, (batch_size, 1, 1))
            }
            if embedding>0:
                batch_dataset['res_idx'] = np.tile(res_type, (batch_size, 1))
            else:
                batch_dataset['res_type'] = np.tile(res_type, (batch_size, 1, 1))
            batch_dataset = BatchDataset(batch_dataset, device)

            if n_steps == split+1:
                complete_epoch = True

            yield batch_dataset, complete_epoch
            split += 1

        # Deal with the part smaller than a batch_size
        left_len = n_data % batch_size
        if left_len != 0 and drop_last:
            continue

        elif left_len != 0 and not drop_last:
            data_batch = data[split * batch_size:]

            batch_dataset = {
                'structures': data_batch,
                'meta': {'smearing': smearing,
                         'gaussian': gaussian,
                         'gaussian_margin':gaussian_margin,
                         'gaussian_normalize': gaussian_normalize,
                         'gaussian_factor': gaussian_factor},
                'tor_type': np.tile(tor_type, (data_batch.shape[0], 1, 1))
            }
            if embedding>0:
                batch_dataset['res_idx'] = np.tile(res_type, (data_batch.shape[0], 1))
            else:
                batch_dataset['res_type'] = np.tile(res_type, (data_batch.shape[0], 1, 1))
            batch_dataset = BatchDataset(batch_dataset, device)

            yield batch_dataset, True

        if shuffle:
            np.random.shuffle(data)


