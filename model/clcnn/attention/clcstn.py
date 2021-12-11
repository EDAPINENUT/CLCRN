import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from ..graphconv import GraphConv

class MSTGCNBlock(nn.Module):
    r"""An implementation of the Multi-Component Spatial-Temporal Graph
    Convolution block `_
    
    Args:
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filters (int): Number of Chebyshev filters.
        nb_time_filters (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
    """
    def __init__(self, in_channels: int, conv_ker, max_view: int, nb_chev_filter: int,
                 nb_time_filter: int, time_strides: int):
        super(MSTGCNBlock, self).__init__()
        
        self.conv_ker = conv_ker
        self.conv = GraphConv(
            input_dim=in_channels,
            output_dim=nb_chev_filter,                                     
            max_view=max_view,
            conv=conv_ker
            )
        self._time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter,
                                    kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
                                    
        self._residual_conv = nn.Conv2d(in_channels, nb_time_filter,
                                        kernel_size=(1, 1), stride=(1, time_strides))
                                        
        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self.nb_time_filter = nb_time_filter
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass with a single MSTGCN block.

        Arg types:
            * X (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in). 
            * edge_index (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * X (PyTorch FloatTensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """

        num_of_timesteps, batch_size, num_of_vertices, in_channels = X.shape
            
        X_tilde = X
        X_tilde = F.relu(self.conv(x=X_tilde))# num_of_timesteps, batch_size, num_of_vertices, nb_time_filter
        X_tilde = X_tilde.permute(1, 3, 2, 0) # batch_size, nb_time_filter, num_of_vertices, num_of_timesteps
    
        X_tilde = self._time_conv(X_tilde) 
        X = self._residual_conv(X.permute(1, 3, 2, 0))
        X = self._layer_norm(F.relu(X + X_tilde).permute(0, 3, 2, 1))
        X = X.permute(1, 0, 2, 3)
        return X


class CLCSTN(nn.Module):
    r"""An implementation of the Multi-Component Spatial-Temporal Graph Convolution Networks, a degraded version of ASTGCN.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional 
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_
    
    Args:
        
        nb_block (int): Number of ASTGCN blocks in the model.
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filter (int): Number of Chebyshev filters.
        nb_time_filter (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
    """
    def __init__(self, nb_block: int, input_dim: int, output_dim: int, max_view: int, nb_chev_filter: int,
                 nb_time_filter: int, time_strides: int, seq_len: int, horizon: int,  conv_ker, embed_dim=None,
                 **model_kwargs):
        super(CLCSTN, self).__init__()

        self.horizon = horizon
        self.output_dim = output_dim
        self.conv_ker = conv_ker
        if embed_dim is not None:
            self.input_dim = input_dim + embed_dim * 2
        self._blocklist = nn.ModuleList([MSTGCNBlock(self.input_dim, self.conv_ker, max_view, nb_chev_filter, nb_time_filter, time_strides)])

        self._blocklist.extend([MSTGCNBlock(nb_time_filter, self.conv_ker, max_view, nb_chev_filter, nb_time_filter, 1) for _ in range(nb_block-1)])

        self._final_conv = nn.Conv2d(int(seq_len/time_strides), seq_len*output_dim, kernel_size=(1, nb_time_filter))

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        
        T_in, B, N_nodes, F_in = X.shape
        F_out = self.output_dim
        T_out = self.horizon

        for block in self._blocklist:
            X = block(X)

        X = self._final_conv(X.permute(1, 0, 2, 3))
        X = X[:, :, :, -1]
        X = X.reshape(B, T_out, F_out, N_nodes)
        X = X.permute(1, 0, 3, 2)
        return X
