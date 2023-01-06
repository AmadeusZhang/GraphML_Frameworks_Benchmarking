from constants import *
from typing import Any
from collections import Counter
import os
import networkx as nx

class ModelRunner:
    data = None
    models: dict[Models, Any] = {}
    bk_dataset: Datasets = None
    verbose = False
    G = None
    num_classes = None

    def load(self,bk_dataset: Datasets):
        this_folder_path = os.path.dirname(os.path.abspath(__file__))
        label = 'class_label'
        file_path = os.path.join(this_folder_path,
                                 "data",
                                 "planetoid_split" if bk_dataset.planetoid_split else "random_split",
                                 f'{bk_dataset.name.lower()}.graphml')
        G = nx.read_graphml(file_path)
        class_labels = bk_dataset.mappings
        classes = convert_attr_to_number(graph=G, attr_name=label, class_labels=class_labels)
        self.num_classes = len(classes)
        self.G = G
        self.bk_dataset = bk_dataset
    def load_and_convert(self, bk_dataset: Datasets):
        self.load(bk_dataset)
        self.data = None
        self.bk_dataset = bk_dataset

    def create_models(self):
        print("Create model, edit this to create all the models in your framework")
        self.models[Models.GCN] = None

    def train_model(self, model: Models, split_num: int = 0):
        print("To implement on each framework, train model on split with split_num")

    def reset_weights(self, model: Models):
        print("To implement on each framework, reset weights of model")

    def __init__(self, verbose=False):
        self.verbose = verbose

    def run_all(self):
        num_iter_planetoid = 2  # numer of iteration for each model in planetoid split
        num_random_splits = 2  # number of random splits to test

        # Train with planetoid split
        print("Running with planetoid split") if self.verbose else None
        for dataset in Datasets:
            print(f"Loading dataset {dataset.lower}") if self.verbose else None
            self.load_and_convert(dataset.with_planetoid_split())
            self.create_models()
            for model in Models:
                for iter in range(num_iter_planetoid): #todo average over iterations
                    print(f"Training model {model} on dataset {dataset}") if self.verbose else None
                    self.train_model(model, 0)
                    print(f"Resetting weights for model {model}") if self.verbose else None
                    self.reset_weights(model)

        # Train with random split
        for dataset in Datasets:
            self.load_and_convert(dataset.with_planetoid_split())
            self.create_models()
            for model in Models:
                for split_num in range(num_random_splits): #todo average over iterations
                    print(f"Training model {model} with split {split_num}") if self.verbose else None
                    self.train_model(model)
                    print(f"Resetting weights for model {model}") if self.verbose else None
                    self.reset_weights(model)


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
