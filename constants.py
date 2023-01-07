from enum import Enum

citeseer_mappings = {
    3: 'DB',
    2: 'IR',
    4: 'Agents',
    1: 'ML',
    5: 'HCI',
    0: 'AI',
}

cora_mappings = {
    3: 'Neural_Networks',
    4: 'Probabilistic_Methods',
    2: 'Genetic_Algorithms',
    0: 'Theory',
    5: 'Case_Based',
    1: 'Reinforcement_Learning',
    6: 'Rule_Learning'
}

pubmed_mappings = {  # Planetoid:Ours
    2: '2',
    1: '3',
    0: '1'
}


# List of datasets used in our experiments
# Lowercase dataset names,camelcase dataset names, mappings between public splits numbers and labels
class Datasets(Enum):
    CORA = 'cora', 'Cora', cora_mappings
    CITESEER = 'citeseer', 'CiteSeer', citeseer_mappings
    PUBMED = 'pubmed', 'PubMed', pubmed_mappings

    def __init__(self, lower, CamelCase, mappings, planetoid_split: bool = True):
        self.lower: str = lower
        self.CamelCase = CamelCase
        self.mappings = mappings
        self.planetoid_split = planetoid_split

    def with_planetoid_split(self):
        self.planetoid_split = True
        return self

    def with_random_split(self):
        self.planetoid_split = False
        return self

    def class_names(self):
        return self.mappings.values()

class Frameworks(Enum):
    PYG = 'pyg'
    STELLARGRAPH = 'stellargraph'
    DGL = 'dgl'

    def __init__(self, lower):
        self.lower = lower


class HyperparameterGiver:

    def __init__(self):
        '''
        The unified early stopping criterion training stops if the total validation
        loss (loss on the data plus regularization loss) does not improve for 50 epochs
        '''
        self.patience = 50
        self.epochs = 100000
        self.learning_rate = None
        self.dropout = None
        self.hidden_size = None
        self.l2_regularization = None
        self.seed = 42  # Todo check the actual seed from the paper


class gcnHG(HyperparameterGiver):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.01
        self.dropout = 0.8
        self.hidden_size = 64
        self.l2_regularization = 0.001


class gatHG(HyperparameterGiver):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.01
        self.dropout = 0.6
        self.hidden_size = 64
        self.l2_regularization = 0.01
        self.att_dropout = 0.3


class graphsageHG(HyperparameterGiver):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.dropout = 0.4
        self.hidden_size = 32
        self.l2_regularization = 0.1


class Models(Enum):
    GCN = 'gcn', gcnHG()
    GAT = 'gat', gatHG()
    GRAPHSAGE = 'sage', graphsageHG()

    def __init__(self, lower, params):
        self.lower = lower
        self.params = params
