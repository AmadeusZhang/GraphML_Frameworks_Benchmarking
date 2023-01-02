from model_runner import *
import torch_geometric.utils.convert as conv
from torch_geometric.nn.sequential import Sequential
from pyg_models.pyg_utils import *
from pyg_models.summary import ssummary
from metrics import *


device = "cuda" if torch.cuda.is_available() else "cpu"


class PygModelRunner(ModelRunner):

    def load_and_convert(self, bk_dataset: Datasets):
        self.load(bk_dataset)
        label = 'class_label'

        # Delete unused graph attributes
        delete_graph_attributes(self.G)
        print(self.G.graph, type(self.G.graph))

        print("Loading x...")

        nodes_to_load = [attr for attr in list(self.G.nodes(data=True))[0][1].keys() if 'w.' in attr]
        data: Data = conv.from_networkx(self.G, group_node_attrs=nodes_to_load,
                                        group_edge_attrs=None)

        data.x = data.x.float()

        data.y = data[label].long()
        # one hot encode labels
        data.y = torch.nn.functional.one_hot(data.y.long(), num_classes=self.num_classes).to(torch.float)

        split_num = 0

        data.train_mask = data[f'train_mask.{split_num}'].bool()
        data.test_mask = data[f'test_mask.{split_num}'].bool()
        data.val_mask = data[f'validation_mask.{split_num}'].bool()

        print(
            f"Number of nodes in train={data.train_mask.sum().item()}, test={data.test_mask.sum().item()}, val={data.val_mask.sum().item()}")

        print(data)
        print(data.x)
        self.data = data


    def create_models(self):
        create_gcn(self)
        create_graph_sage(self)
        create_gat(self)

    def reset_weights(self, model: Models):
        self.create_models()  # todo more performant version

    def train_model(self, model: Models, split_num: int = 0):
        self.data.train_mask = self.data[f'train_mask.{split_num}'].bool()
        self.data.test_mask = self.data[f'test_mask.{split_num}'].bool()
        self.data.val_mask = self.data[f'validation_mask.{split_num}'].bool()

        pyg_model = self.models[model]
        # Set parameters
        torch.manual_seed(model.params.seed)
        # Copy data to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = self.data.to(device)

        history = train(pyg_model, data, model.params)
        plot_history(history)

        # Test the model and save the results.
        pyg_model.eval()
        out = pyg_model(data.x, data.edge_index)
        y_cat = data.y[data.test_mask].detach().clone()
        display_and_save(framework=Frameworks.PYG,
                         dataset_name=self.bk_dataset,
                         model_name=model.lower,
                         predictions=out[data.test_mask].cpu().detach().numpy(),
                         y=y_cat.cpu().detach().numpy(),
                         class_names=self.bk_dataset.class_names(),
                         folder_name='metrics',
                         exec_ms=0)


def create_gcn(model_runner: PygModelRunner):
    from torch_geometric.nn.models import GCN
    hidden_dim = Models.GCN.params.hidden_size
    dropout = Models.GCN.params.dropout
    model = Sequential('x, edge_index', [
        (GCN(in_channels=model_runner.data.num_features, hidden_channels=hidden_dim, num_layers=2,
             out_channels=model_runner.data.y.shape[1],
             dropout=dropout),
         'x, edge_index -> x'),
        torch.nn.LogSoftmax(dim=1)
    ])
    print(
        ssummary(model, model_runner.data.x.cpu(), model_runner.data.edge_index.cpu(), max_depth=50, leaf_module=None))
    model_runner.models[Models.GCN] = model


def create_graph_sage(model_runner: PygModelRunner):
    from torch_geometric.nn.models import GraphSAGE
    hidden_dim = Models.GRAPHSAGE.params.hidden_size
    dropout = Models.GRAPHSAGE.params.dropout
    model = Sequential('x, edge_index', [
        (GraphSAGE(in_channels=model_runner.data.num_features, hidden_channels=hidden_dim, num_layers=2,
                   out_channels=model_runner.data.y.shape[1],
                   dropout=dropout),
         'x, edge_index -> x'),
        torch.nn.LogSoftmax(dim=1)
    ])
    print(
        ssummary(model, model_runner.data.x.cpu(), model_runner.data.edge_index.cpu(), max_depth=50, leaf_module=None))
    model_runner.models[Models.GRAPHSAGE] = model


def create_gat(model_runner: PygModelRunner):
    from torch_geometric.nn.models import GAT
    hidden_dim = Models.GAT.params.hidden_size
    dropout = Models.GAT.params.dropout
    model = Sequential('x, edge_index', [
        (GAT(in_channels=model_runner.data.num_features, hidden_channels=hidden_dim, num_layers=2,
             out_channels=model_runner.data.y.shape[1],
             dropout=dropout),
         'x, edge_index -> x'),
        torch.nn.LogSoftmax(dim=1)
    ])
    print(
        ssummary(model, model_runner.data.x.cpu(), model_runner.data.edge_index.cpu(), max_depth=50, leaf_module=None))
    model_runner.models[Models.GAT] = model


# main function
if __name__ == '__main__':
    pyg_runner: PygModelRunner = PygModelRunner()
    pyg_runner.run_all()
