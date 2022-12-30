import networkx as nx

import torch
from torch import Tensor
from torch_geometric.data import Data
from collections import Counter
from metrics import *
from constants import HyperparameterGiver


# Converts a module to a module usable by pytorch geometric sequential
# def seq(module):
#     return (module, 'x, edge_index -> x')

# load the graph from the graphml file
def load_graph(file_path):
    G = nx.read_graphml(file_path)
    return G


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


class History:
    loss: list[float]
    acc: list[float]
    val_loss: list[float]
    val_acc: list[float]

    def __init__(self):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def print_last(self):
        epoch = len(self.loss)
        print(
            f"Epoch {epoch}: loss={self.loss[-1]:.4f}, acc={self.acc[-1]:.4f}, val_loss={self.val_loss[-1]:.4f}, val_acc={self.val_acc[-1]:.4f}")


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


def plot_history(history: History):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.loss, label="loss")
    ax[0].plot(history.val_loss, label="val_loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history.acc, label="acc")
    ax[1].plot(history.val_acc, label="val_acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()

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


# Converts a string attribute to a number and returns the list of all the possible values of that attribute
def convert_attr_to_number(graph, attr_name: str, class_labels: dict[int:str]):
    classes = []
    if len(class_labels) == 0:
        for id in graph.nodes:
            classes.append(graph.nodes[id][attr_name])
        print(f"No class labels, printing counter of classes {Counter(classes)}")
        return Counter(classes)

    mappings = {w: k for k, w in class_labels.items()}
    mappings = {**mappings, **{i: i for i in range(len(class_labels))}}
    for id in graph.nodes:
        graph.nodes[id][attr_name] = mappings[graph.nodes[id][attr_name]]
    return class_labels.values()


import torch


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
