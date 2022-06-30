import os
import numpy as np
import yaml

import torch
from torch.optim import Adam

from autoeisd.data import torsion_loader, split_data
from autoeisd.models import RecurrentModelResidue
from autoeisd.train import Trainer
from autoeisd.utils import residue_sc_marker

drkseq = 'MEAIAKHDFSATADDELSFRKTQILKILNMEDDSNWYRAELDGKEGLIPSNYIEMKNHD'
asyn_seq = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAV\
VTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'

IDP_SEQ = drkseq

settings_path = 'yml_settings/torsion_recurrent_residue.yml'
settings = yaml.safe_load(open(settings_path, "r"))

device = torch.device(settings['general']['device'])

# data
root = settings['data']['root'].strip()
data = np.load(os.path.join(root, 'bbsc.npy'))
train, val, _ = split_data(data[:7373], train_size=5000, val_size=500, seed=1)
smearing = (settings['data']['start_val'],
            settings['data']['stop_val'],
            settings['data']['ohe_size'])

train_gen = torsion_loader(train,
                           IDP_SEQ,
                           residue_based=True,
                           embedding=3,
                           smearing=smearing,
                           gaussian=settings['data']['gaussian'],
                           gaussian_margin=settings['data']['gaussian_margin'],
                           gaussian_normalize=settings['data']['gaussian_normalize'],
                           gaussian_factor=settings['data']['gaussian_factor'],
                           batch_size=settings['training']['tr_batch_size'],
                           device=device,
                           shuffle=settings['training']['shuffle']) 

val_gen = torsion_loader(val,
                         IDP_SEQ,
                         residue_based=True,
                         embedding=3,
                         smearing=smearing,
                         gaussian=settings['data']['gaussian'],
                         gaussian_margin=settings['data']['gaussian_margin'],
                         gaussian_normalize=settings['data']['gaussian_normalize'],
                         gaussian_factor=settings['data']['gaussian_factor'],
                         batch_size=settings['training']['val_batch_size'],
                         device=device)

# model
model = RecurrentModelResidue(recurrent=settings['model']['recurrent'],
                          filter_in=settings['data']['ohe_size'],
                          n_filter_layers=settings['filter']['n_layer'],
                          filter_out=settings['filter']['out'],
                          filter_drop=settings['filter']['dropout'],
                          rec_stack_size=settings['model']['rec_stack_size'],
                          rec_neurons_num=settings['model']['rec_neurons_num'],
                          rec_dropout=settings['model']['rec_dropout'],
                          embed_out=settings['filter']['embedding'],
                          embed_in=3)


# optimizer
# encoder_trainable_params = filter(lambda p: p.requires_grad, encoder.parameters())
optimizer = Adam(model.parameters(),
                 lr=settings['training']['lr_enc'],
                 weight_decay=settings['training']['weight_decay_enc'])

# loss
def compute_loss(x, x_hat, seq=IDP_SEQ):
    # mask out padded torsions
    # x_hat shape -1, seq_len*8, bins
    idx = []
    num = residue_sc_marker(seq[1:]) + 3
    for n in range(len(seq)-1):
         idx.extend(list(np.arange(num[n])+n*8))
    # remove the end backbones torsions
    idx.pop(-4+num[-1])
    # apply the mask
    x = x[:, idx, :]
    x_hat = x_hat[:, idx, :] 
    
    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2])
    loginp = torch.nn.functional.log_softmax(inp, dim=1)
    recon_loss = -(target * loginp).sum() / loginp.size()[0] 
    print('recon_loss:', recon_loss.detach().cpu().numpy())

    return recon_loss

# training
trainer = Trainer(
    sequence=IDP_SEQ,
    model=model,
    loss_fn=compute_loss,
    optimizer=optimizer,
    device=device,
    yml_path=settings['general']['me'],
    driver_path=settings['general']['driver'],
    output_path=settings['general']['output'],
    lr_scheduler=settings['training']['lr_scheduler'],
)
trainer.print_layers()

trainer.train(
    mode=settings['general']['output'][1],
    train_gen=train_gen,
    val_gen=val_gen,
    epochs=settings['training']['n_epochs'],
    clip_grad=settings['training']['clip_grad'],
    tr_batch_size=settings['training']['tr_batch_size'],
    val_batch_size = settings['training']['val_batch_size'],
    kld_alpha = settings['training']['KLD_alpha'],
)

print('done!')
