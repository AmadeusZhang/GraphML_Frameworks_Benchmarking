from constants import *
from typing import Any


class ModelRunner:
    data = None
    models: dict[Models, Any] = {}
    bk_dataset: Datasets = None
    verbose = False

    def load_and_convert(self, bk_dataset: Datasets):
        self.data = None

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
