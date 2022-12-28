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
        self.dropout = 0.3
        self.hidden_size = 64
        self.l2_regularization = 0.01
        self.att_dropout = 0.6


class graphsageHG(HyperparameterGiver):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.dropout = 0.4
        self.hidden_size = 32
        self.l2_regularization = 0.1
