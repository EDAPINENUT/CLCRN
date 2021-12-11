import numpy as np
import torch
import torch.nn as nn
from ..seq2seq import Seq2SeqAttrs
from .clcstn import CLCSTN
from .. clconv import CLConv
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CLCSTNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, loc_info, sparse_idx, geodesic, angle_ratio, logger=None, **model_kwargs):
        '''
        Conditional Local Convolution Recurrent Network, implemented based on DCRNN,
        Args:
            loc_info (torch.Tensor): location infomation of each nodes, with the shape (node_num, location_dim). For sphercial signals, location_dim=2.
            sparse_idx (torch.Tensor): sparse_idx with the shape (2, node_num * nbhd_num).
            geodesic (torch.Tensor): geodesic distance between each point and its neighbors, with the shape (node_num * nbhd_num), corresponding to sparse_idx.
            angle_ratio (torch.Tensor): the defined angle ratio contributing to orientation density, with the shape (node_num * nbhd_num), corresponding to sparse_idx.
            model_kwargs (dict): Other model args see the config.yaml.
        '''
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, geodesic, angle_ratio, **model_kwargs)
        self.register_buffer('node_embeddings', nn.Parameter(torch.randn(self.node_num, self.embed_dim), requires_grad=True))
        self.feature_embedding = nn.Linear(self.input_dim, self.embed_dim)
        self._logger = logger
        self.conv_ker = CLConv(
            self.location_dim, 
            self.sparse_idx,
            self.node_num,
            self.lck_structure, 
            loc_info, 
            self.angle_ratio, 
            self.geodesic,
            self.max_view
        )
        self.network = CLCSTN(
            nb_block=self.block_num,
            nb_chev_filter=self.hidden_units,
            nb_time_filter=self.hidden_units,
            time_strides=int(self.seq_len/2),
            conv_ker=self.conv_ker,
            **model_kwargs
        )

    def embedding(self, inputs):
        # inputs: b, l, n, f
        # outputs: b, l, n, e*2
        batch_size, seq_len, node_num, feature_size = inputs.shape
        feature_emb = self.feature_embedding(inputs)
        node_emb = self.node_embeddings[None, None, :, :].expand(batch_size, seq_len, node_num, self.embed_dim)
        return torch.cat([feature_emb, node_emb, inputs], dim=-1)

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor, input_dim)
        :param labels: shape (horizon, batch_size, num_sensor, output_dim)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.node_num * self.output_dim)
        """
        embedding = self.embedding(inputs)
        outputs = self.network(embedding)
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs
