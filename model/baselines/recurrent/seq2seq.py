import torch

class Seq2SeqAttrs:
    def __init__(self, sparse_idx, **model_kwargs):
        self.sparse_idx = sparse_idx
        self.max_view = int(model_kwargs.get('max_view', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.node_num = int(model_kwargs.get('node_num', 6))
        self.layer_num = int(model_kwargs.get('layer_num', 2))
        self.rnn_units = int(model_kwargs.get('rnn_units', 32))
        self.input_dim = int(model_kwargs.get('input_dim', 2))
        self.output_dim = int(model_kwargs.get('output_dim', 2))
        self.seq_len = int(model_kwargs.get('seq_len', 12))
        self.embed_dim = int(model_kwargs.get('embed_dim', 16))
        self.horizon = int(model_kwargs.get('horizon', 16))
