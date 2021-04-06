import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.model import NNBase
from a2c_ppo_acktr.utils import init

class IAMBase(NNBase):
    def __init__(self, num_inputs, dset=None, fnn_hidden_layer=512, fnn_last_layer=256, rnn_last_layer=128):

        if dset == None:
            rnn_input_size = num_inputs
        else:
            rnn_input_size = len(dset)

        super(IAMBase, self).__init__(True, rnn_input_size, rnn_last_layer)

        self.dset = dset
        self._output_size = fnn_last_layer+rnn_last_layer

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.fnn = nn.Sequential(
            init_(nn.Linear(num_inputs, fnn_hidden_layer)), nn.Tanh(),
            init_(nn.Linear(fnn_hidden_layer, fnn_last_layer)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(self._output_size, 1))

        self.gru_tanh = nn.Tanh()

        self.train()
    
    @property
    def output_size(self):
        return self._output_size

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        
        y_rnn, rnn_hxs = self._forward_gru(x[..., self.dset], rnn_hxs, masks)
        y_rnn = self.gru_tanh(y_rnn)

        y_fnn = self.fnn(x)

        y = torch.cat((y_fnn, y_rnn), dim=-1)

        return self.critic_linear(y), y, rnn_hxs
    
class GRUBase(NNBase):
    def __init__(self, num_inputs, fnn_hidden_layer=640, rnn_last_layer=128):
        super(GRUBase, self).__init__(True, num_inputs, rnn_last_layer)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.fnn = nn.Sequential(
            init_(nn.Linear(num_inputs, fnn_hidden_layer)), nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(rnn_last_layer, 1))

        self.gru_tanh = nn.Tanh()

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        y = self.fnn(x)
        
        y, rnn_hxs = self._forward_gru(y, rnn_hxs, masks)
        y = self.gru_tanh(y)

        return self.critic_linear(y), y, rnn_hxs