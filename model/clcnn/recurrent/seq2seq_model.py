import numpy as np
import torch
import torch.nn as nn
from ..clconv import CLConv
from ..seq2seq import Seq2SeqAttrs
from .encoder import EncoderModel
from .decoder import DecoderModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CLCRNModel(nn.Module, Seq2SeqAttrs):
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
        self.encoder_model = EncoderModel(sparse_idx, geodesic, angle_ratio, self.conv_ker, **model_kwargs)
        self.decoder_model = DecoderModel(sparse_idx, geodesic, angle_ratio, self.conv_ker, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def get_kernel(self):
        return self.conv_ker

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):

        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.node_num, self.output_dim))
        go_symbol = go_symbol.to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                    decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

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
        encoder_hidden_state = self.encoder(embedding)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs
