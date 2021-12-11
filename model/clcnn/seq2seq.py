import torch

class Seq2SeqAttrs:
    def __init__(self, sparse_idx, angle_ratio, geodesic, **model_kwargs):
        self.sparse_idx = sparse_idx
        self.max_view = int(model_kwargs.get('max_view', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.node_num = int(model_kwargs.get('node_num', 6))
        self.layer_num = int(model_kwargs.get('layer_num', 2))
        self.rnn_units = int(model_kwargs.get('rnn_units', 32))
        self.input_dim = int(model_kwargs.get('input_dim', 2))
        self.output_dim = int(model_kwargs.get('output_dim', 2))
        self.seq_len = int(model_kwargs.get('seq_len', 12))
        self.lck_structure = model_kwargs.get('lckstructure', [4,8])
        self.embed_dim = int(model_kwargs.get('embed_dim', 16))
        self.location_dim = int(model_kwargs.get('location_dim', 16))
        self.horizon = int(model_kwargs.get('horizon', 16))
        self.hidden_units = int(model_kwargs.get('hidden_units', 16))
        self.block_num = int(model_kwargs.get('block_num', 2))
        angle_ratio = torch.sparse.FloatTensor(
            self.sparse_idx, 
            angle_ratio, 
            (self.node_num,self.node_num)
            ).to_dense() 
        self.angle_ratio = angle_ratio + torch.eye(*angle_ratio.shape).to(angle_ratio.device)
        self.geodesic =  torch.sparse.FloatTensor(
            self.sparse_idx, 
            geodesic, 
            (self.node_num,self.node_num)
            ).to_dense()