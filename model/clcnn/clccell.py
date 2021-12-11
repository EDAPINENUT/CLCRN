import torch
import torch.nn as nn
import torch.nn.functional as F
from .graphconv import *

class CLCRNCell(torch.nn.Module):
    def __init__(
        self, 
        num_units, 
        sparse_idx, 
        max_view, 
        node_num, 
        num_feature, 
        conv_ker, 
        num_embedding,
        nonlinearity='tanh'
        ):

        super().__init__()

        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._node_num = node_num
        self._num_feature = num_feature
        self._num_units = num_units
        self._max_view = max_view
        self._sparse_idx = sparse_idx
        self._num_embedding = num_embedding
        self.conv_ker = conv_ker

        self.ru_gconv = GraphConv(
            input_dim=self._num_embedding,
            output_dim=self._num_units * 2, 
            max_view=self._max_view,
            conv=conv_ker
            )
        self.c_gconv = GraphConv(
            input_dim=self._num_embedding,
            output_dim=self._num_units,                                     
            max_view=self._max_view,
            conv=conv_ker
            )
        

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, node_num, input_dim) 
        :param hx: (B, node_num, rnn_units)
        :param t: (B, num_time_feature)
        :return
        - Output: A `3-D` tensor with shaconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerpe `(B, node_num, rnn_units)`.
        """

        conv_in_ru = self._concat(inputs, hx)
        value = torch.sigmoid(self.ru_gconv(conv_in_ru))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._node_num, self._num_units))
        u = torch.reshape(u, (-1, self._node_num, self._num_units))
        conv_in_c = self._concat(inputs, r*hx)
        c = self.c_gconv(conv_in_c)

        if self._activation is not None:
            c = self._activation(c)
        new_state = u * hx + (1.0 - u) * c
        
        return new_state

    @staticmethod
    def _concat(x, x_):
        return torch.cat([x, x_], dim=2)

 