import torch
import torch.nn as nn
from ..clccell import CLCRNCell
from ..seq2seq import Seq2SeqAttrs

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, sparse_idx, angle_ratio, geodesic, conv_ker, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, sparse_idx, angle_ratio, geodesic, **model_kwargs)
        self.conv = conv_ker
        self.clgru_layers = self.init_clgru_layers()
        self.projection_layer = nn.Linear(self.input_dim + 2*self.embed_dim, self.rnn_units)
        
    def init_clgru_layers(self):
        module_list = []
        for i in range(self.layer_num):
            if i == 0:
                input_dim = self.input_dim
            else:
                input_dim = self.rnn_units
            module_list.append(CLCRNCell(
                self.rnn_units, 
                self.sparse_idx, 
                self.max_view, 
                self.node_num, 
                input_dim, 
                self.conv,
                self.rnn_units * 2
            ))
        return nn.ModuleList(module_list)

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.node_num * self.input_dim)
        :param hidden_state: (layer_num, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, __, _  = inputs.size()
        inputs = self.projection_layer(inputs)
        if hidden_state is None:
            hidden_state = torch.zeros((self.layer_num, batch_size, self.node_num, self.rnn_units))
            hidden_state = hidden_state.to(inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.clgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state            

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow