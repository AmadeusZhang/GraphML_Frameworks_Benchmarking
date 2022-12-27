from collections import Counter

import networkx as nx
import numpy as np
from torch_geometric.datasets import Planetoid
import torch as torch

import gnnbench.data.io as io
import gnnbench.data.make_dataset as make_dataset
from numpy import load
from metrics import BkDataset
import torch_geometric.utils.convert as conv
import os

thisfolderpath = os.path.dirname(os.path.abspath(__file__))
npzfolderpath = os.path.join(thisfolderpath, 'data/npz')

def features_to_dict(feature_name:str,features_list):
    return {f"{feature_name}.{i}": val for i, val in enumerate(features_list)}
def index_array_to_bool(index_array, length):
    bool_array = np.zeros(length, dtype=bool)
    bool_array[index_array] = True
    return bool_array

def convert_random_split(name:BkDataset):
    name = name.value['lower']
    print(f"\nConverting {name}")
    npzfilepath = os.path.join(npzfolderpath, f'{name}.npz')
    data = load(npzfilepath)
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])
    dataset = io.load_dataset(npzfolderpath + "\\" + name)

    print('num_nodes', dataset.num_nodes())
    print('num_edges', dataset.num_edges())

    dataset_standard : io.SparseGraph = dataset.standardize()
    # for attr in dir(dataset_standard):
    #     if not attr.startswith('__'):
    #         print(attr, getattr(dataset_standard, attr))

    print('num_nodes', dataset_standard.num_nodes())
    print('num_edges', dataset_standard.num_edges())
    print('attr_names', dataset_standard.attr_names)
    print('class_names', dataset_standard.class_names)
    print('is_directed', dataset_standard.is_directed())
    print('labels', dataset_standard.labels)
    #min of labels is
    print('min of labels', np.min(dataset_standard.labels))
    print('max of labels', np.max(dataset_standard.labels))
    #One hot of lables using pytorch
    lables_tensor = torch.tensor(dataset_standard.labels)
    num_classes = np.max(dataset_standard.labels)+1
    labels_one_hot = torch.nn.functional.one_hot(lables_tensor.long(), num_classes=num_classes).numpy()

    adj_matrix,features,labels = dataset_standard.unpack()

    num_nodes = dataset_standard.num_nodes()

    num_splits = 100
    print(f"Creating {num_splits} random splits...")
    train_masks = np.zeros(shape=(num_nodes, num_splits), dtype=bool)
    val_masks = np.zeros(shape=(num_nodes, num_splits), dtype=bool)
    test_masks = np.zeros(shape=(num_nodes, num_splits), dtype=bool)
    for i in range(num_splits):
        random_state = np.random.RandomState(i)
        split = make_dataset.get_train_val_test_split(random_state,
                                                      labels_one_hot,
                                                      train_examples_per_class=20, val_examples_per_class=30,
                                                      test_examples_per_class=None,)
        #print(f"split {i}")

        train_masks[:,i] = index_array_to_bool(split[0], len(labels))
        val_masks[:,i] = index_array_to_bool(split[1], len(labels))
        test_masks[:,i] = index_array_to_bool(split[2], len(labels))


    print("train_masks shape", train_masks.shape)
    print("sum of train_masks", np.sum(train_masks, axis=0))

    print(train_masks)
    features = features.toarray()

    print(features)
    print(adj_matrix.shape)
    #node0 = features_to_dict('w', features[0,:])
    #print(node0)
    print(features.shape)

    #print(features_to_dict('class_labels',labels))
    features = [features_to_dict('w', features[i, :].tolist()) for i in range(dataset_standard.num_nodes())]
    for i in range(dataset_standard.num_nodes()):
        features[i]['class_label'] = str(dataset_standard.class_names[labels[i]] if dataset_standard.class_names is not None else labels[i])
        features[i] = {**features[i], **features_to_dict('train_mask', train_masks[i,:].tolist())}
        features[i] = {**features[i], **features_to_dict('validation_mask', val_masks[i,:].tolist())}
        features[i] = {**features[i], **features_to_dict('test_mask', test_masks[i,:].tolist())}


    features_and_id = [(i,features[i]) for i in range(dataset_standard.num_nodes())]
    print(features_and_id[0])

    G = nx.from_scipy_sparse_array(adj_matrix)
    G.add_nodes_from(features_and_id)

    if dataset_standard.node_names is not None:
        nx.relabel_nodes(G, {i: dataset_standard.node_names[i] for i in range(dataset_standard.num_nodes())}, copy=False)

    #remove edge weights
    for u, v, d in G.edges(data=True):
        del d['weight']

    print(list(G.nodes(data=True))[2])
    # print(type(G.nodes['100701']['w.10']))
    # print(type(G.nodes['100701']['class_label']))
    # print(type(G.nodes['100701']['train_mask.0']))
    graphml_path = os.path.join(thisfolderpath,'data/random_split', f'{name}.graphml')
    print(f'Saving {name} graphml to {graphml_path}')
    nx.write_graphml(G, graphml_path)

def convert_public_split(dataset_info:BkDataset):
    dataset_name = dataset_info.value['CamelCase']
    print(f"\nConverting {dataset_name}")
    pyg_dataset = Planetoid(f"/tmp/{dataset_name}", name=dataset_name, split='public')
    print(pyg_dataset[0].edge_index.shape)
    # generate_masks('cora',train_mask=dataset[0].train_mask,test_mask=dataset[0].test_mask,val_mask=dataset[0].val_mask)
    G1 = conv.to_networkx(data=pyg_dataset[0], node_attrs=['x', 'y', 'train_mask', 'test_mask', 'val_mask'],to_undirected='upper')
    # sparse_mat = nx.to_scipy_sparse_matrix(G1)
    # print(sparse_mat.nnz)
    #G1.remove_edges_from([(i,i) for i in range(len(G1.nodes))])
    print(len(G1.edges)/2)
    mappings = dataset_info.value['mappings']

    #Print classes if there are no class names in dataset
    if len(mappings) == 0:
        print(pyg_dataset[0])
        classes = []
        for id in G1.nodes:
            classes.append(G1.nodes[id]['y'])
        print(f"No class labels, printing counter of classes {Counter(classes)}")
        return Counter(classes)

    print('Working on graphml')
    print("Edges: ", len(G1.edges))
    for id, data in G1.nodes(data=True):
        for i, v in enumerate(data['x']):
            data[f"w.{i}"] = float(v) if dataset_name == 'PubMed' else int(v)
        del data['x']
        data['class_label'] = mappings[int(data.pop('y'))] #rename y to class_label
        data['train_mask.0'] = data.pop('train_mask')
        data['test_mask.0'] = data.pop('test_mask')
        data['validation_mask.0'] = data.pop('val_mask')
        # data['y']=class_labels[int(data['y'])]
        # data['subject'] = data.pop('y') #rename y to subject
    #
    # classes = []
    # for id in G1.nodes:
    #     if G1.nodes[id]['train_mask']:
    #         classes.append(G1.nodes[id]['class_label'])
    # print(f"No class labels, printing counter of classes {Counter(classes)}")

    print(G1.nodes[0])
    name = dataset_info.value['lower']
    graphml_path = os.path.join(thisfolderpath,'data/planetoid_split', f'{name}.graphml')
    print(f'Saving {dataset_name} graphml to {graphml_path}')
    nx.write_graphml(G1, graphml_path)


convert_random_split(BkDataset.CORA)
convert_random_split(BkDataset.PUBMED)
convert_random_split(BkDataset.CITESEER)

convert_public_split(BkDataset.CORA)
convert_public_split(BkDataset.PUBMED)
convert_public_split(BkDataset.CITESEER)
