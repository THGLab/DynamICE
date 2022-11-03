import os
import numpy as np
import yaml

import torch
from torch.optim import Adam

from dynamice.data import torsion_loader, split_data
from dynamice.models import RecurrentModel
from dynamice.train import Trainer
from dynamice.utils import residue_sc_marker

drkseq = 'MEAIAKHDFSATADDELSFRKTQILKILNMEDDSNWYRAELDGKEGLIPSNYIEMKNHD'
asynseq = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAV\
VTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'

IDP_SEQ = drkseq

settings_path = 'torsion_recurrent.yml'
settings = yaml.safe_load(open(settings_path, "r"))

device = torch.device(settings['general']['device'])

# data
root = settings['data']['root'].strip()
data = np.load(os.path.join(root, 'bbsc.npy'))
train, val, _ = split_data(data[:1000], train_size=500, val_size=100, seed=1)
smearing = (settings['data']['start_val'],
            settings['data']['stop_val'],
            settings['data']['ohe_size'])

train_gen = torsion_loader(train,
                           IDP_SEQ,
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
                         embedding=3,
                         smearing=smearing,
                         gaussian=settings['data']['gaussian'],
                         gaussian_margin=settings['data']['gaussian_margin'],
                         gaussian_normalize=settings['data']['gaussian_normalize'],
                         gaussian_factor=settings['data']['gaussian_factor'],
                         batch_size=settings['training']['val_batch_size'],
                         device=device)

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


# optimizer
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
    idx.pop(2-num[-1])
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
    train_gen=train_gen,
    val_gen=val_gen,
    epochs=settings['training']['n_epochs'],
    clip_grad=settings['training']['clip_grad'],
    tr_batch_size=settings['training']['tr_batch_size'],
    val_batch_size = settings['training']['val_batch_size'])

print('done!')
