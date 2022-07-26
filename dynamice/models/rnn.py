import torch
from torch import nn
from dynamice.models import MLP


class RecurrentModel(nn.Module):

    def __init__(self,
                 recurrent,
                 filter_in,
                 n_filter_layers,
                 filter_drop,
                 filter_out,
                 rec_stack_size,
                 rec_neurons_num,
                 rec_dropout,
                 embed_out,
                 embed_in
                 ):
        """
        GRU Recurrent units for the language model.
        Model: recurrent model with torsion angle, type and

        Parameters
        ----------
        recurrent: str
            'gru', 'lstm'
        n_filter_layers: int
        filter_out: int
        neurons: list
        latent_dimension; int
        """
        super(RecurrentModel, self).__init__()
        self.rec_stack_size = rec_stack_size
        self.rec_neurons_num = rec_neurons_num
        self.res_embedding = nn.Embedding(20**embed_in, embed_out)       

        # filter and transform torsion angles
        self.filter = MLP(8*filter_in+embed_out, filter_out,
                          n_layers=n_filter_layers,
                          activation=nn.ReLU(), dropout=filter_drop)

        # Reduce dimension up to second last layer of Encoder
        if recurrent == 'gru':
            self.res_recurrent = nn.GRU(input_size=filter_out,
                                    hidden_size=rec_neurons_num,
                                    num_layers=rec_stack_size,
                                    batch_first=True,
                                    dropout=rec_dropout)
        elif recurrent == 'lstm':
            self.res_recurrent = nn.LSTM(input_size=filter_out,
                                     hidden_size=rec_neurons_num,
                                     num_layers=rec_stack_size,
                                     batch_first=True,
                                     dropout=rec_dropout)
        
        # attention layer
        self.tor_recurrent = nn.LSTM(input_size=filter_out,
                            hidden_size=rec_neurons_num,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)
        
        # linear mapping
        self.linear_omega = nn.Linear(rec_neurons_num, filter_in)
        self.linear_phi = nn.Linear(rec_neurons_num, filter_in)
        self.linear_psi = nn.Linear(rec_neurons_num, filter_in)
        self.linear_out = nn.Linear(rec_neurons_num, filter_in)
        self.tor_filter = MLP(rec_neurons_num+filter_in+8, filter_out,
                               n_layers=2, activation=nn.ReLU())

        

    def init_hidden(self, batch_size=1, stack_size=None):
        weight = next(self.parameters())
        if stack_size is None: stack_size = self.rec_stack_size
        return weight.new_zeros(stack_size, batch_size,
                                self.rec_neurons_num)

    def init_cell(self, stack_size=None, batch_size=1):
        weight = next(self.parameters())
        if stack_size is None: stack_size = self.rec_stack_size
        return weight.new_zeros(stack_size, batch_size,
                                self.rec_neurons_num)
    
    def torsion_recurrent(self, angle, tortype, x, hidden):
        '''
        Recurrent unit for torsion angles within a residue

        Parameters
        ----------
        angle : torch.tensor
            last torsion angle
        x : torch.tensor
            hidden from RNN

        Returns
        -------
        x : torch.tensor
            current torsion angle to generate
        '''
        
        # concatenate torsion with output from rnn
        x = torch.cat([angle.unsqueeze(1), tortype.unsqueeze(1), x], dim=-1) # batch, seq, 8+hidden+bins
        x = self.tor_filter(x)
        
        # Get results of encoder network
        x, hidden = self.tor_recurrent(x, hidden) #batch, seq, hidden_size

        # last layer
        if tortype[0].argmax() == 1:
            x = self.linear_psi(x)
        else:
            x = self.linear_out(x) #shape (batch, seq, bins)
        return x, hidden
        

    def forward(self, tor_angle, res_type, tor_type, hidden):
        """
        Pass through the language model.

        Parameters
        ----------
        tor_angle: torch.tensor
            (batch, seq*8, ohe_size)

        res_type: torch.tensor
            (batch, seq)
        """
        last_tor_angle = tor_angle[:, :8, :]
        b, seq, f = last_tor_angle.shape
        # concatenate 8 torsion angles
        x = last_tor_angle.view(b, seq//8, f*8) # batch,seq,8*filter_out
        res_type = self.res_embedding(res_type)
        
        # concatenate all features
        x = torch.cat([x, res_type], dim=2)
        # filter
        x = self.filter(x)
        # Get results of encoder network
        x, hidden = self.res_recurrent(x, hidden) #batch, seq, hidden_size
        
        outlist = []
        outlist.append(self.linear_omega(x))
        outlist.append(self.linear_phi(x))
        in_hidden = (self.init_hidden(b, 1), self.init_hidden(b, 1))
        
        for n in range(6):
            new_tor, in_hidden = self.torsion_recurrent(tor_angle[:, n+9], 
                                             tor_type[:, n+1], 
                                             x, in_hidden)
            outlist.append(new_tor)
        
        torsions = torch.cat(outlist, dim=-1)   # batch, seq, n_bins*8
        torsions = torsions.view(b, seq, f)  # batch, seq*8, n_bins

        return torsions, hidden

    def generate(self, tor_angle, res_type, hidden):
        """
        Pass throught the language model.

        Parameters
        ----------
        tor_angle: torch.tensor
            (1, seq*3, n_bins)

        res_type: torch.tensor
            (1, seq*3, n_bins)

        Returns
        -------
        torch.tensor: probabilities
        torch.tensor: hidden state for the next residue

        """
        b, seq, f = tor_angle.shape
        # concatenate 8 torsion angles
        x = tor_angle.view(b, seq//8, f*8)
        res_type = self.res_embedding(res_type)
        # concatenate all features
        x = torch.cat([x, res_type], dim=2)
        # filter
        x = self.filter(x)
        # Get results of encoder network
        x, hidden = self.res_recurrent(x, hidden)
        self.rnn_out = x

        # last layer
        omega = self.linear_omega(x)
        phi = self.linear_phi(x)

        # softmax
        phi = nn.functional.softmax(phi, dim=-1)
        omega = nn.functional.softmax(omega, dim=-1)
        
        outlist = []
        outlist.append(omega)
        outlist.append(phi)
        torsions = torch.cat(outlist, dim=-1)   # batch, seq, n_bins*2
        torsions = torsions.reshape(b, 2, -1)  # batch, seq*2, n_bins
        # need to call torsion recurrent separately to sample the rest of angles

        return torsions, hidden
