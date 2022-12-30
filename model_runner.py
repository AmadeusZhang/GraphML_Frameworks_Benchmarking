from metrics import *


class ModelRunner:
    data = None
    models = {}
    bk_dataset: BkDataset = None
    verbose = False

    def load_and_convert(self,bk_dataset: BkDataset):
        print("Loading and converting data, edit this to load and add data in your framework")
        self.data = None

    def create_model(self):
        print("Create model, edit this to create all the models in your framework")
        self.models['model_name'] = None

    def train_model(self):
        print("Train model")
        print(self.models, self.data)

    def __init__(self, verbose=False):
        self.verbose = verbose

    def run_all(self):
        self.load_and_convert(BkDataset.CORA)
        self.create_model()
        self.train_model()