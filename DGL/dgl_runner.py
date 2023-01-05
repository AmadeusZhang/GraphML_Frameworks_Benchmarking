from model_runner import *
import dgl
from constants import Datasets
import torch
from pyg_models.pyg_utils import History, plot_history
from metrics import display_and_save
import torch.nn as nn
import torch.nn.functional as F


class DglModelRunner(ModelRunner):

    def load_and_convert(self, bk_dataset: Datasets):
        self.load(bk_dataset)
        label = 'class_label'

        # Delete unused graph attributes
        print(self.G.graph, type(self.G.graph))
        print(list(self.G.nodes(data=True))[0])

        print("Loading x...")

        node = list(self.G.nodes(data=True))[0][1]
        nodes_to_load = list(node.keys())
        data: dgl.DGLGraph = dgl.from_networkx(self.G, node_attrs=nodes_to_load)
        # print(data)

        graph = dgl.from_networkx(self.G)

        graph.ndata['feat'] = data.ndata['w.0'].reshape(1, -1).T
        for i in nodes_to_load:
            if 'w.' in i and i != 'w.0':
                tessorTuple = (graph.ndata['feat'], data.ndata[i].reshape(1, -1).T)
                graph.ndata['feat'] = torch.cat(tessorTuple, 1)
            elif 'train_mask.' in i:
                graph.ndata[i] = data.ndata[i]
            elif 'validation_mask.' in i:
                graph.ndata[f'val_mask.{i.split(sep=".")[1]}'] = data.ndata[i]
            elif 'test_mask.' in i:
                graph.ndata[i] = data.ndata[i]

        graph.ndata['label'] = data.ndata[label]
        # print(graph.ndata['feat'])

        graph = dgl.add_self_loop(graph)

        print(
            f"Number of nodes in train={graph.ndata['train_mask.0'].sum().item()},"
            f"test={graph.ndata['test_mask.0'].sum().item()},"
            f"val={graph.ndata['val_mask.0'].sum().item()}")

        print(graph)
        # print(graph.ndata)

        self.data = graph

    def create_models(self):
        create_gcn(self)
        create_graph_sage(self)
        create_gat(self)

    def reset_weights(self, model: Models):
        self.create_models()  # todo more performant version

    def train_model(self, model: Models, split_num: int = 0):
        features = self.data.ndata['feat']
        labels = self.data.ndata['label']
        masks = self.data.ndata[f'train_mask.{split_num}'], self.data.ndata[f'val_mask.{split_num}'], self.data.ndata[f'test_mask.{split_num}']

        pyg_model = self.models[model]
        # Set parameters
        torch.manual_seed(model.params.seed)
        # Copy data to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = self.data.to(device)

        history = self.train(data, features, labels, masks, pyg_model)
        # plot_history(history)

        # Test the model and save the results.
        logits = pyg_model(self.data, features.float())
        test_mask = masks[2]
        test_logits = logits[test_mask]
        test_labels = labels[test_mask]
        one_hot_y_test = F.one_hot(test_labels)
        display_and_save(framework=Frameworks.DGL,
                         dataset_name=self.bk_dataset,
                         model_name=model.lower,
                         predictions=test_logits.cpu().detach().numpy(),
                         y=one_hot_y_test.cpu().detach().numpy(),
                         class_names=self.bk_dataset.class_names(),
                         folder_name='metrics',
                         exec_ms=0)

    def train(self, g, features, labels, masks, model) -> History:
        # define train/val samples, loss function and optimizer
        history = History()
        train_mask = masks[0]
        val_mask = masks[1]
        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=model.metrics.learning_rate)

        # training loop
        for epoch in range(200):
            model.train()
            logits = model(g, features.float())
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            val_loss = loss_fcn(logits[val_mask], labels[val_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = self.evaluate(g, features, labels, train_mask, model)
            val_acc = self.evaluate(g, features, labels, val_mask, model)

            # print(loss, type(loss))
            history.loss.append(loss)
            history.acc.append(acc)
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)
            history.print_last()

        return history

    def evaluate(self, g, features, labels, mask, model):
        model.eval()
        with torch.no_grad():
            logits = model(g, features.float())
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)


def create_gcn(model_runner: DglModelRunner):
    from models.gcn.pygcn import GCN
    # pytorch gcn model
    g = model_runner.data
    hidden_dim = Models.GCN.params.hidden_size
    model = GCN(g.ndata['feat'].shape[1], hidden_dim, model_runner.num_classes)
    # print(
    #     ssummary(model, model_runner.data.x.cpu(), model_runner.data.edge_index.cpu(), max_depth=50, leaf_module=None))
    print(model)
    model_runner.models[Models.GCN] = model


def create_graph_sage(model_runner: DglModelRunner):
    from models.graphsage.pygraphsage import SAGE
    g = model_runner.data
    model = SAGE(g.ndata['feat'].shape[1], 256, model_runner.num_classes)
    print(model)
    model_runner.models[Models.GRAPHSAGE] = model


def create_gat(model_runner: DglModelRunner):
    from models.gat.pygat import GAT
    g = model_runner.data
    model = GAT(g.ndata['feat'].shape[1], 8, model_runner.num_classes, heads=[8, 1])
    print(model)
    model_runner.models[Models.GAT] = model


# main function
if __name__ == '__main__':
    dgl_runner: DglModelRunner = DglModelRunner()
    # dgl_runner.load_and_convert(Datasets.CORA)
    # dgl_runner.create_models()
    # dgl_runner.train_model(Models.GAT, 0)
    dgl_runner.run_all()

