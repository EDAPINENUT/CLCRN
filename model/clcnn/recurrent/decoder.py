import torch
import torch.nn as nn
from ..clccell import CLCRNCell
from ..seq2seq import Seq2SeqAttrs

class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, sparse_idx, geodesic, angle_ratio, conv_ker, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, sparse_idx, geodesic, angle_ratio, **model_kwargs)
        self.conv = conv_ker
        self.clgru_layers = self.init_clgru_layers()
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)

    def init_clgru_layers(self):
        module_list = []
        for i in range(self.layer_num):
            if i == 0:
                input_dim = self.output_dim
            else:
                input_dim = self.rnn_units
            module_list.append(CLCRNCell(
                            self.rnn_units, 
                            self.sparse_idx, 
                            self.max_view, 
                            self.node_num, 
                            input_dim, 
                            self.conv,
                            input_dim + self.rnn_units
                        ))        
        return nn.ModuleList(module_list)

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.node_num * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.node_num * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.clgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output)
        output = projected.reshape(-1, self.node_num, self.output_dim)

        return output, torch.stack(hidden_states)