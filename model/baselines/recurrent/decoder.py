import torch
import torch.nn as nn
from .seq2seq import Seq2SeqAttrs

class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, sparse_idx, conv, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, sparse_idx, **model_kwargs)
        self.conv = conv
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.edge_weights = torch.ones_like(self.sparse_idx[0]).float()

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
        for layer_num, layer in enumerate(self.conv):
            arg_dict = {'X': output, 'edge_index': self.sparse_idx, 'edge_weight':self.edge_weights, 'H':hidden_state[layer_num]}
            next_hidden_state = layer(**arg_dict)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output)
        output = projected.reshape(-1, self.node_num, self.output_dim)

        return output, torch.stack(hidden_states)