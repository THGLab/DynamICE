import numpy as np
import os
from sklearn.model_selection import train_test_split

def split_data(data, train_size, val_size, 
               seed=0, save_path=None):
    # mode: L/ L+E/ L+H+E structures, currently takes idx range
    # asyn L: [:2282], L+E+: [2282: 4903], L+E+H+: [4903:]
    # drk L: [:3438], L+E+: [3438: 7372], L+E+H+: [7372:]

    indices = np.arange(data.shape[0])
    train_idx, els = train_test_split(indices, train_size=train_size, random_state=seed)
    val_idx, test_idx = train_test_split(els, train_size=val_size, random_state=seed)
    train_bbsc = data[train_idx]
    val_bbsc = data[val_idx]
    test_bbsc = data[test_idx]
    print(len(train_idx), len(val_idx), len(test_idx))
    print(test_bbsc[0, 8:10])
    if save_path is not None:
        #np.save(os.path.join(save_path, 'train_idx.npy'), train_idx)
        np.save(os.path.join(save_path, 'test_idx.npy'), test_idx)
        #np.save(os.path.join(save_path, 'val_idx.npy'), val_idx)
        #np.save(os.path.join(save_path, 'train_bbsc.npy'), train_bbsc)
        np.save(os.path.join(save_path, 'test_bbsc.npy'), test_bbsc)
        #np.save(os.path.join(save_path, 'val_bbsc.npy'), val_bbsc)
    return train_bbsc, val_bbsc, test_bbsc
