import os
import numpy as np
import pandas as pd
import torch
import time
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from dynamice.utils.utility import (
  generate_tor_marker, 
  residue_sc_marker,
)
from dynamice.utils.definitions import sidechain_res

#backbone bond angles stats, torsion_position_marker
omega_pos = np.nan
phi_pos = np.nan
psi_pos = np.nan
chi1_pos = np.nan
chi2_pos = np.nan
chi3_pos = np.nan
chi4_pos = np.nan
chi5_pos = np.nan


def populate_tor_marker(sequence):
    # torsion angle marker 
    global omega_pos, phi_pos, psi_pos, chi1_pos, chi2_pos, chi3_pos, chi4_pos, chi5_pos
    seq = generate_tor_marker(sequence)
    first = 1 + sidechain_res[sequence[0]]
    omega_pos = np.argwhere(seq[first:] == 0).flatten()
    phi_pos = np.argwhere(seq[first:] == 1).flatten()
    psi_pos = np.argwhere(seq[first:] == 2).flatten()
    chi1_pos = np.argwhere(seq[first:] == 3).flatten()
    chi2_pos = np.argwhere(seq[first:] == 4).flatten()
    chi3_pos = np.argwhere(seq[first:] == 5).flatten()
    chi4_pos = np.argwhere(seq[first:] == 6).flatten()
    chi5_pos = np.argwhere(seq[first:] == 7).flatten()

def compare_diff_soft(x_degree, x_hat_degree, pos, tol=360):
    
    diff1 = np.abs(x_degree[:, pos] - x_hat_degree[:, pos])
    diff2 = np.abs(x_degree[:, pos] - x_hat_degree[:, pos] - tol)
    diff3 = np.abs(x_degree[:, pos] - x_hat_degree[:, pos] + tol)
    diff = np.min(np.stack([diff1, diff2, diff3], axis=2), axis=2)
    mae = np.mean(diff)
    return mae

def compute_recon_quality(x, x_hat, bb=False):

    bin_size = 360.0 / x.shape[-1]
    x_indices = x.argmax(-1)
    x_hat_indices = x_hat.argmax(-1)

    x_degree = x_indices*bin_size + bin_size/2
    x_hat_degree = x_hat_indices*bin_size + bin_size/2

    # omega distance
    omega_mae = compare_diff_soft(x_degree, x_hat_degree, omega_pos)
    # phi distance
    phi_mae = compare_diff_soft(x_degree, x_hat_degree, phi_pos)
    # psi distance
    psi_mae = compare_diff_soft(x_degree, x_hat_degree, psi_pos)

    # chi_distance
    chi1_mae = compare_diff_soft(x_degree, x_hat_degree, chi1_pos)
    chi2_mae = compare_diff_soft(x_degree, x_hat_degree, chi2_pos)
    chi3_mae = compare_diff_soft(x_degree, x_hat_degree, chi3_pos)
    chi4_mae = compare_diff_soft(x_degree, x_hat_degree, chi4_pos)
    if len(chi5_pos) == 0:
        chi5_mae = np.zeros_like(phi_mae)
    else:
        chi5_mae = compare_diff_soft(x_degree, x_hat_degree, chi5_pos)
    return [omega_mae, phi_mae, psi_mae, chi1_mae, chi2_mae,
            chi3_mae, chi4_mae, chi5_mae]

def compute_recon_quality_residue(x, x_hat, seq):

    # mask out padded angles, x shape (batch, seq*8, bins)
    idx = []
    num = residue_sc_marker(seq[1:]) + 3
    for n in range(len(seq)-1):
        idx.extend(list(np.arange(num[n])+n*8))
    # remove the end backbones torsions
    idx.pop(-num[-1]-1)
    x = x[:, idx, :]
    x_hat = x_hat[:, idx, :]

    return compute_recon_quality(x, x_hat)


def compute_recon_quality_backbone(x, x_hat):

    # x shape (batch, seq*3, bins)
    ntors  = x.shape[1]
    # omega distance
    omega_mae = compare_diff_soft(x_degree, x_hat_degree, np.arange(0, ntors, 3))
    # phi distance
    phi_mae = compare_diff_soft(x_degree, x_hat_degree, np.arange(1, ntors, 3))
    # psi distance
    psi_mae = compare_diff_soft(x_degree, x_hat_degree, np.arange(2, ntors, 3))

    return [omega_mae, phi_mae, psi_mae]

class Trainer:
    """
    Parameters
    ----------
    model: tuple
    optimizer: tuple
    """

    def __init__(self,
                 sequence,
                 model,
                 loss_fn,
                 optimizer,
                 device,
                 yml_path,
                 driver_path,
                 output_path,
                 lr_scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.sequence = sequence

        path_iter = output_path[1]
        out_path = os.path.join(output_path[0], 'training_%i'%path_iter)
        while os.path.exists(out_path):
            path_iter+=1
            out_path = os.path.join(output_path[0], 'training_%i'%path_iter)
        os.makedirs(out_path)
        self.output_path = out_path

        script_out = os.path.join(self.output_path, 'run_scripts')
        os.makedirs(script_out)
        shutil.copyfile(yml_path, os.path.join(script_out,os.path.basename(yml_path)))
        shutil.copyfile(driver_path, os.path.join(script_out,driver_path))

        # learning rate scheduler
        self.handle_scheduler(lr_scheduler, optimizer)
        self.lr_scheduler = lr_scheduler

        # checkpoints
        self.epoch = 0        # number of epochs of any steps that model has gone through so far
        self.data_point = 0   # number of data points that model has seen so far
        self.log = {'epoch': [], 'steps':[], 'loss': [],
                    'val_loss': [],
                    'val_omega_mae': [], 'val_phi_mae': [], 'val_psi_mae': [],
                    'val_chi1_mae': [], 'val_chi2_mae': [], 'val_chi3_mae': [], 
                    'val_chi4_mae': [], 'val_chi5_mae': [], 
                    'lr_enc': [], 'time': []
                    }
        self.best_val = -10

    def handle_scheduler(self, lr_scheduler, optimizer):

        if lr_scheduler[0] == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                           mode=lr_scheduler[5],
                                           patience = lr_scheduler[2],
                                           factor = lr_scheduler[3],
                                           min_lr = lr_scheduler[4])
        elif lr_scheduler[0] == 'decay':
            lambda1 = lambda epoch: np.exp(-epoch * lr_scheduler[1])
            scheduler = LambdaLR(optimizer=optimizer,
                                  lr_lambda=lambda1)
        else:
            raise NotImplementedError('scheduler "%s" is not implemented yet.'%lr_scheduler[0])

        self.scheduler = scheduler

    def print_layers(self):
        total_n_params = 0
        #for model in self.model:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                if len(param.shape) > 1:
                    total_n_params += param.shape[0] * param.shape[1]
                else:
                    total_n_params += param.shape[0]
        print('\n total trainable parameters: %i\n' % total_n_params)

    def store_checkpoint(self, input, steps):
        self.log['epoch'].append(self.epoch)
        self.log['steps'].append(steps)
        self.log['loss'].append(input[0])
        self.log['val_loss'].append(input[1])
        self.log['lr_enc'].append(input[2])
        self.log['time'].append(input[3])
         
        if self.mode == 'backbone':
            self.log['val_omega_mae'].append(input[4][0])
            self.log['val_phi_mae'].append(input[4][1])
            self.log['val_psi_mae'].append(input[4][2])
            print("[%d, %3d] loss: %.6f; val_loss: %.6f; val_omega_mae: %.6f; val_phi_mae: %.6f; \
              val_psi_mae: %.6f; lr_enc: %.8f; epoch_time: %.3f" %  
              (self.epoch, steps, input[0], input[1], input[4][0], input[4][1], input[4][2], input[2], input[3]))
            pd.DataFrame(self.log).to_csv(os.path.join(self.output_path, 'log.csv'), index=False)
            return

        if self.mode == 'sidechain':
            self.log['val_chi1_mae'].append(input[4][3])
            self.log['val_chi2_mae'].append(input[4][4])
            self.log['val_chi3_mae'].append(input[4][5])
            self.log['val_chi4_mae'].append(input[4][6])
            self.log['val_chi5_mae'].append(input[4][7])
            pd.DataFrame(self.log).to_csv(os.path.join(self.output_path, 'log.csv'), index=False)
            print("[%d, %3d] loss: %.6f; val_loss: %.6f; val_x1_mae: %.6f; val_x2_mae: %.6f; val_x3_mae: %.6f; \
                  val_x4_mae: %.6f; val_x5_mae: %.6f; lr_enc: %.8f; epoch_time: %.3f" %  
              (self.epoch, steps, input[0], input[1], input[4][0], input[4][1], input[4][2], input[4][3], 
               input[4][4], input[2], input[3]))
            return
        
        self.log['val_omega_mae'].append(input[4][0])
        self.log['val_phi_mae'].append(input[4][1])
        self.log['val_psi_mae'].append(input[4][2])
        self.log['val_chi1_mae'].append(input[4][3])
        self.log['val_chi2_mae'].append(input[4][4])
        self.log['val_chi3_mae'].append(input[4][5])
        self.log['val_chi4_mae'].append(input[4][6])
        self.log['val_chi5_mae'].append(input[4][7])
        pd.DataFrame(self.log).to_csv(os.path.join(self.output_path, 'log.csv'), index=False)
        print("[%d, %3d] loss: %.6f; val_loss: %.6f; val_omega_mae: %.6f; val_phi_mae: %.6f; \
              val_psi_mae: %.6f; val_x1_mae: %.6f; val_x2_mae: %.6f; val_x3_mae: %.6f; \
                  val_x4_mae: %.6f; val_x5_mae: %.6f; lr_enc: %.8f; epoch_time: %.3f" %  
              (self.epoch, steps, input[0], input[1], input[4][0], input[4][1], input[4][2], input[4][3], 
               input[4][4], input[4][5], input[4][6], input[4][7], input[2], input[3]))

    def _optimizer_to_device(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        return optimizer


    def train(self, train_gen, val_gen, epochs, clip_grad,
                                           tr_batch_size, val_batch_size):
        """
        The main function to train model for the given number of epochs.
        The implementation allows for resuming the training with different data and number of epochs.

        Parameters
        ----------
        mode
        train_gen
        val_gen
        epochs
        clip_grad

        Returns
        -------

        """
        model = self.model
        model.to(self.device)  
        populate_tor_marker(self.sequence)
        assert omega_pos is not np.nan
        self.best_val = 1e5
        self.mode = 'all'
        optimizer = self._optimizer_to_device(self.optimizer)

        running_val_quality = []
        for _ in range(epochs):
            t0 = time.time()

            # record total number of epochs
            self.epoch += 1

            # training
            model.train()

            running_loss = 0.0
            step = 0
            completed_epoch = False
            while not completed_epoch:
                hidden = model.init_hidden(tr_batch_size)
                hidden = (hidden, hidden)
                train_batch, completed_epoch = next(train_gen)
                torsion_angle = train_batch.structures
                torsion_angle_out = torsion_angle[:, 8:, :]
                residue_type = train_batch.res_type[:, 1:]
                torsion_type = train_batch.tor_type

                out = torch.zeros_like(torsion_angle_out, device=self.device)
                # decoding from RNN N times, where N is the length of sequence
                for n in range(len(self.sequence)-1):
                    angle_in = torsion_angle[:, 8*n:8*n+15, :]
                    res_type = residue_type[:, n].unsqueeze(1)
                    out_ohe, hidden = model(angle_in, res_type, torsion_type, hidden)
                    out[:, 8*n:8*n+8, :] = out_ohe
                # compute loss
                loss = self.loss_fn(torsion_angle_out, out)

                # back propogation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                if clip_grad>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

                # collect loss
                step += 1
                step_loss = loss.item()
                running_loss += step_loss

                del train_batch
            running_loss /= step

            # validation
            vallist = []
            tor_quality_list = []
            if val_gen is not None:
                model.eval()
                with torch.no_grad():
                    val_step = 0
                    completed_epoch = False
                    while not completed_epoch:
                        hidden = model.init_hidden(val_batch_size)
                        hidden = (hidden, hidden)
                        val_batch, completed_epoch = next(val_gen)
                        torsion_angle = val_batch.structures
                        torsion_angle_out = torsion_angle[:, 8:, :]
                        residue_type = val_batch.res_type[:, 1:]
                        torsion_type = val_batch.tor_type

                        out = torch.zeros_like(torsion_angle_out, device=self.device)
                        # decoding from RNN N times, where N is the length of sequence
                        for n in range(len(self.sequence)-1):
                            angle_in = torsion_angle[:, 8*n:8*n+15, :]
                            res_type = residue_type[:, n].unsqueeze(1)
                            out_ohe, hidden = model(angle_in, res_type, torsion_type, hidden)
                            out[:, 8*n:8*n+8, :] = out_ohe
                        # store data
                        #in_struct.append(torsion_angle_out.data.cpu().numpy())
                        #out_struct.append(out.data.cpu().numpy())

                        vallist.append(
                            self.loss_fn(torsion_angle_out, out).detach().cpu().numpy()
                        )
                        tor_quality_list.append(compute_recon_quality_residue(
                            torsion_angle_out.detach().cpu().numpy(),
                            out.detach().cpu().numpy(), self.sequence
                        ))

                        val_step+=1

                        del val_batch

                valloss = np.mean(vallist)
                val_mae = np.mean(tor_quality_list, axis=0)

            else:
                valloss = 1e5

            # best model
            if self.best_val > valloss:
                self.best_val = valloss
                torch.save(model.state_dict(), os.path.join(self.output_path, 'best_model.tar'))

            # learning rate decay
            if self.lr_scheduler[0] == 'plateau':
                running_val_quality.append(valloss)
                if len(running_val_quality) > self.lr_scheduler[1]:
                    running_val_quality.pop(0)
                accum_val_quality = np.mean(running_val_quality)
                self.scheduler.step(accum_val_quality)
            elif self.lr_scheduler[0] == 'decay':
                self.scheduler.step()

            # checkpoint
            for i, param_group in enumerate(self.scheduler.optimizer.param_groups):
                lr = float(param_group["lr"])

            self.store_checkpoint((running_loss, valloss,
                                   lr, time.time()-t0, val_mae), step)
