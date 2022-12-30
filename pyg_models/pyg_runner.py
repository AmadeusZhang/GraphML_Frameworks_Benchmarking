from model_runner import ModelRunner
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric.utils.convert as conv
import torch
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn.sequential import Sequential
import torch.nn.functional as F
import training_hyperparameters as hp
from pyg_utils import *
import os
import json
from summary import ssummary
from metrics import *

this_folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(this_folder_path, "../data", "planetoid_split", "cora.graphml")
label = 'class_label'
device = "cuda" if torch.cuda.is_available() else "cpu"
num_features = 1433
classes = None
num_classes = None


class PygModelRunner(ModelRunner):

    def load_and_convert(self, bk_dataset: BkDataset):
        print("Loading and converting data")
        G = load_graph(file_path)
        class_labels = bk_dataset.mappings
        classes = convert_attr_to_number(graph=G, attr_name=label, class_labels=class_labels)
        num_classes = len(classes)

        # Delete unused graph attributes
        delete_graph_attributes(G)
        print(G.graph, type(G.graph))

        print("Loading x...")

        data: Data = conv.from_networkx(G, group_node_attrs=[f'w.{i}' for i in range(num_features)],
                                        group_edge_attrs=None)

        data.x = data.x.float()

        data.y = data[label].long()
        # one hot encode labels
        data.y = torch.nn.functional.one_hot(data.y.long(), num_classes=num_classes).to(torch.float)

        split_num = 0

        data.train_mask = data[f'train_mask.{split_num}'].bool()
        data.test_mask = data[f'test_mask.{split_num}'].bool()
        data.val_mask = data[f'validation_mask.{split_num}'].bool()

        print(
            f"Number of nodes in train={data.train_mask.sum().item()}, test={data.test_mask.sum().item()}, val={data.val_mask.sum().item()}")

        print(data)
        print(data.x)
        self.data = data

    def create_model(self):
        create_gcn(self)
        create_graph_sage(self)
        create_gat(self)

    def train_model(self):
        class_labels = {}
        model = self.models['gcn']
        params = hp.gcnHG()

        train_and_eval(model=model.to(torch.device(device)),data=self.data, labels=class_labels.values(),parameters=params)


def create_gcn(model_runner: PygModelRunner):
    from torch_geometric.nn.models import GCN
    hidden_dim = 16
    model = Sequential('x, edge_index', [
        (GCN(in_channels=model_runner.data.num_features, hidden_channels=hidden_dim, num_layers=2,
             out_channels=model_runner.data.y.shape[1],
             dropout=0.5),
         'x, edge_index -> x'),
        torch.nn.LogSoftmax(dim=1)
    ])
    # model = GCN(in_channels=in_features, hidden_channels=16,act='relu',num_layers=2,out_channels=num_classes,dropout=0.5)
    print(
        ssummary(model, model_runner.data.x.cpu(), model_runner.data.edge_index.cpu(), max_depth=50, leaf_module=None))
    model_runner.models['gcn'] = model


def create_graph_sage(model_runner: PygModelRunner):
    from torch_geometric.nn.models import GraphSAGE
    model = Sequential('x, edge_index', [
        (GraphSAGE(in_channels=model_runner.data.num_features, hidden_channels=16, num_layers=2,
                   out_channels=model_runner.data.y.shape[1],
                   dropout=0.5),
         'x, edge_index -> x'),
        torch.nn.LogSoftmax(dim=1)
    ])
    print(
        ssummary(model, model_runner.data.x.cpu(), model_runner.data.edge_index.cpu(), max_depth=50, leaf_module=None))
    model_runner.models['graph_sage'] = model


def create_gat(model_runner: PygModelRunner):
    from torch_geometric.nn.models import GAT
    model = Sequential('x, edge_index', [
        (GAT(in_channels=model_runner.data.num_features, hidden_channels=8, num_layers=2,
             out_channels=model_runner.data.y.shape[1],
             dropout=0.5),
         'x, edge_index -> x'),
        torch.nn.LogSoftmax(dim=1)
    ])
    print(
        ssummary(model, model_runner.data.x.cpu(), model_runner.data.edge_index.cpu(), max_depth=50, leaf_module=None))
    model_runner.models['gat'] = model


# main function
if __name__ == '__main__':
    pyg_runner: PygModelRunner = PygModelRunner()

    pyg_runner.run_all()
