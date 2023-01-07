import networkx as nx

import torch
from torch import Tensor
from torch_geometric.data import Data
from collections import Counter
from utils import *
from constants import HyperparameterGiver
from utils import History


# Converts a module to a module usable by pytorch geometric sequential
# def seq(module):
#     return (module, 'x, edge_index -> x')

def apply_regularization(model: torch.nn.Module, loss, l2_reg: float):
    for name, params in model.state_dict().items():
        if name.startswith("conv1"):
            loss += l2_reg * params.square().sum() / 2.0
    return loss


def compute_loss_and_val(model, data, loss_fn, l2_reg, mask: Tensor):
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[mask], data.y[mask])
    loss = apply_regularization(model, loss, l2_reg)  # Regularization

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(data.y[mask].argmax(dim=1).cpu().detach().numpy(),
                         out[mask].argmax(dim=1).cpu().detach().numpy())

    return loss, acc


def train_step(model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer, loss_fn, l2_reg: float):
    model.train()
    optimizer.zero_grad()

    loss, acc = compute_loss_and_val(model, data, loss_fn, l2_reg, data.train_mask)

    # Backpropagation
    loss.backward()
    optimizer.step()
    return loss.item(), acc


@torch.no_grad()  # this is a decorator that tells pytorch not to compute gradients
def eval_step(model: torch.nn.Module, data: Data, loss_fn, mask: Tensor):
    model.eval()
    loss, acc = compute_loss_and_val(model, data, loss_fn, 0, mask)
    return loss.item(), acc


def train(model: torch.nn.Module, data: Data, parameters: HyperparameterGiver) -> History:
    history = History()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.learning_rate, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_model = None
    best_epoch = 0
    for epoch in range(parameters.epochs):
        loss, acc = train_step(model, data, optimizer, loss_fn, parameters.l2_regularization)
        val_loss, val_acc = eval_step(model, data, loss_fn, data.val_mask)
        history.loss.append(loss)
        history.acc.append(acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.print_last()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = model.state_dict()
        elif epoch - best_epoch > parameters.patience:
            print(f"Restoring weights from epoch {best_epoch}")
            model.load_state_dict(best_model)
            break
    return history


def delete_graph_attributes(G):
    # delete all the graph attributes, since they are not useful for node classification

    # print(type(node_attrs))
    # print(edge_attrs)

    # graph.graph['node_default'] = True
    # graph.graph['edge_default'] = True

    print(G.graph, type(G.graph))

    if 'node_default' in G.graph:
        del G.graph['node_default']
    if 'edge_default' in G.graph:
        del G.graph['edge_default']

    # if 'edge_index' in G.graph:
    #     del G.graph['edge_index']

    attrs = [attr for attr, _ in G.graph.items()]

    for attr in attrs:
        del G.graph[attr]


def compareData(data1, data2):
    # Check if x is the same
    if torch.equal(data1.x, data2.x):
        print("x is the same")
    else:
        print("x is different")
        c = data1.x == data2.x
        with open(f'c.json', 'w') as f:
            json.dump(c.tolist(), f)
        with open(f'data_cora.json', 'w') as f:
            json.dump(data2.x.tolist(), f)
    # Check if edge_index is the same
    if torch.equal(data1.edge_index, data2.edge_index):
        print("edge_index is the same")
    else:
        print("edge_index is different")
    # Check if y is the same
    if torch.equal(data1.y, data2.y):
        print("y is the same")
    else:
        print("y is different")
        print(data1.y)
        print(data2.y)
