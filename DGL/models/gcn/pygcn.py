'''
Credits: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/train.py
'''

import dgl.nn.pytorch.conv as dglnn
import torch.nn as nn
import torch.nn.functional as F
from constants import gcnHG


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.metrics = gcnHG()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h