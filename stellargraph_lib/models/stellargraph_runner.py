# Load internal modules
from metrics import *
from constants import *
from model_runner import ModelRunner
from stellargraph_utils import *

# Load external modules
import os
import networkx as nx
import stellargraph
from stellargraph import StellarGraph
from stellargraph.mapper import FullBatchNodeGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GCN, GAT, GraphSAGE
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, losses, Model

from sklearn import preprocessing

this_folder_path = os.path.dirname( os.path.abspath( __file__ ) )

class Data( object ) :
    SG = None
    train_mask = None
    test_mask = None
    val_mask = None

class SGModelRunner( ModelRunner ) :
    
    def load_and_convert( self, bk_dataset: Datasets ) :
        # Load graph
        self.load( bk_dataset )
        label = 'class_label'

        # Delete unused graph attributes
        delete_graph_attributes( self.G )
        print(self.G.graph, type(self.G.graph))

        print("Loading x...")

        # Convert data features
        # for each node, encode weight as a feature
        for node in self.G.nodes():
            enc = [ val for key, val in self.G.nodes[node].items() if key != 'class_label' and key != 'train_mask.0' and key != 'test_mask.0' and key != 'validation_mask.0' ]
            self.G.nodes[node]['feature'] = enc

        sg_data = Data()

        # Save train/test/val masks
        node_subjects = pd.Series( [ self.G.nodes[node]['class_label'] for node in self.G.nodes() ], index = self.G.nodes() )
        train_mask = [ attr["train_mask.0"] for key, attr in list(self.G.nodes(data=True)) ]
        test_mask = [ attr["test_mask.0"] for key, attr in list(self.G.nodes(data=True)) ]
        val_mask = [ attr["validation_mask.0"] for key, attr in list(self.G.nodes(data=True)) ]

        sg_data.train_mask = node_subjects[train_mask]
        sg_data.test_mask = node_subjects[test_mask]
        sg_data.val_mask = node_subjects[val_mask]

        # delete unused attributes
        from copy import deepcopy
        G = deepcopy( self.G )
        for node in G.nodes():
            for key in G.nodes[node].keys():
                if key != 'feature' and key != 'class_label':
                    del self.G.nodes[node][key]
        
        # Convert data to StellarGraph
        SG = StellarGraph( self.G, node_features="feature" )
        print(SG.info())

        sg_data.SG = SG
        self.data = sg_data

    def create_models(self):
        create_gcn(self)
        create_graph_sage(self)
        create_gat(self)


    # TODO: this functions needs to be tested
    def train( self, model: Models, split_num: int = 0 ) :

        # TODO: random split

        sg_model = self.models[model]

        # train/test/val masks are pandas Series with node ids as index and class label as value
        train_mask = self.data.train_mask
        test_mask = self.data.test_mask
        val_mask = self.data.val_mask

        # Create One-Hot encoding for categorical labels
        le = preprocessing.LabelEncoder()
        train_targets = le.fit_transform(train_mask)
        test_targets = le.fit_transform(test_mask)
        val_targets = le.fit_transform(val_mask)

        # create generators
        train_gen = sg_model.generator.flow(train_mask.index, train_targets, shuffle=True)
        test_gen = sg_model.generator.flow(test_mask.index, test_targets)
        val_gen = sg_model.generator.flow(val_mask.index, val_targets)

        # Train model
        history = sg_model.fit(
            train_gen,
            epochs=model.params.epochs, # grape the epochs from the constants file
            validation_data=val_gen,
            verbose=0, # if want to see this progress, set to 1
            shuffle=False
        )

        # Test the model
        y_pred = sg_model.predict(test_gen)
        y_pred = np.argmax(y_pred, axis=1)

        # One-Hot encoding for categorical labels
        y_true = pd.get_dummies(test_targets.values).values

        # Save results
        display_and_save(
            framework=Frameworks.STELLARGRAPH,
            dataset_name=self.bk_dataset,
            model_name=model.lower,
            predictions=y_pred,
            y = y_true,
            class_names=self.bk_dataset.class_names(),
            folder_name='metrics',
            exec_ms=0
        )


def create_gcn( modelrunner: SGModelRunner ) :
    from stellargraph.layer import GCN
    hidden_dim = Models.GCN.params.hidden_size
    dropout = Models.GCN.params.dropout

    # define generator
    generator = FullBatchNodeGenerator( modelrunner.data.SG, method="gcn" )

    # create the GCN model
    gcn = GCN(
        layer_sizes=[hidden_dim, hidden_dim], # num_layers = 2
        activations=["relu", "relu"],
        generator=generator,
        dropout=dropout,
    )
    x_inp, x_out = gcn.in_out_tensors()
    predictions = layers.Dense( units=modelrunner.num_classes, activation="softmax" )( x_out )
    
    # create the keras model
    model = Model( inputs=x_inp, outputs=predictions )
    model.compile(
        optimizer=optimizers.Adam( learning_rate=0.005 ),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )

    # export model summary if needed
    # with open('model_summary.txt', 'w') as f:
    #     model.summary( print_fn=lambda x: f.write(x + '\n') )

    # save model to class
    modelrunner.models[Models.GCN] = model

    # TODO: save generator to class


def create_graph_sage( modelrunner: SGModelRunner ) :
    from stellargraph.layer import GraphSAGE
    hidden_dim = Models.GRAPHSAGE.params.hidden_size
    dropout = Models.GRAPHSAGE.params.dropout

    # define generator
    num_nodes = len( modelrunner.data.SG.nodes() )
    num_samples = [10, 5] # FIXME: or guarantee a motivation for this choice
    generator = GraphSAGENodeGenerator( modelrunner.data.SG, batch_size=num_nodes, num_samples=num_samples )

    # create the GraphSAGE model
    graphsage = GraphSAGE(
        layer_sizes=[hidden_dim, hidden_dim], # num_layers = 2
        activations=["relu", "relu"],
        generator=generator,
        dropout=dropout,
    )
    x_inp, x_out = graphsage.in_out_tensors()
    predictions = layers.Dense( units=modelrunner.num_classes, activation="softmax" )( x_out )

    # create the keras model
    model = Model( inputs=x_inp, outputs=predictions )
    model.compile(
        optimizer=optimizers.Adam( learning_rate=0.005 ),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )

    # export model summary if needed
    # with open('model_summary.txt', 'w') as f:
    #     model.summary( print_fn=lambda x: f.write(x + '\n') )

    # save model to class
    modelrunner.models[Models.GRAPHSAGE] = model


def create_gat( modelrunner: SGModelRunner ) :
    from stellargraph.layer import GAT
    hidden_dim = Models.GAT.params.hidden_size
    dropout = Models.GAT.params.dropout

    # define generator
    generator = FullBatchNodeGenerator( modelrunner.data.SG, method="gat" )

    # create the GAT model
    gat = GAT(
        layer_sizes=[hidden_dim, hidden_dim], # num_layers = 2
        activations=["relu", "relu"],
        attn_heads=8,
        generator=generator,
        in_dropout=dropout,
        attn_dropout=dropout,
        normalize="l2",
    )
    x_inp, x_out = gat.in_out_tensors()
    predictions = layers.Dense( units=modelrunner.num_classes, activation="softmax" )( x_out )

    # create the keras model
    model = Model( inputs=x_inp, outputs=predictions )
    model.compile(
        optimizer=optimizers.Adam( learning_rate=0.005 ),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )

    # export model summary if needed
    # with open('model_summary.txt', 'w') as f:
    #     model.summary( print_fn=lambda x: f.write(x + '\n') )

    # save model to class
    modelrunner.models[Models.GAT] = model
