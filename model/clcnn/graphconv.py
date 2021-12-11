import torch.nn as nn 

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, max_view, conv):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = conv
        self.max_view = max_view
        self.linear = nn.Linear(input_dim * max_view, output_dim)
        # self.norm = nn.LayerNorm(output_dim)#normalization at the channel

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        # x = self.norm(x)
        return x