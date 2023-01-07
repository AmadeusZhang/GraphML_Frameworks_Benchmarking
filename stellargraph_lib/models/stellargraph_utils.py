# Load external modules
import networkx as nx

# Load internal modules
from collections import Counter
from metrics import *
from constants import HyperparameterGiver

def load_graph( file_path: str ) -> nx.Graph :
    G = nx.read_graphml( file_path )
    return G

def convert_attr_to_number( graph, attr_name: str, class_labels: dict[ int:str ]) :
    classes = []
    if len( class_labels ) == 0 :
        for id in graph.nodes :
            classes.append( graph.nodes[ id ][ attr_name ] )
        print( f"No class labels, printing counter of classes {Counter( classes )}" )
        return Counter( classes )

    mappings = { w: k for k, w in class_labels.items() }
    mappings = { **mappings, **{ i: i for i in range( len( class_labels ))}}
    for id in graph.nodes :
        graph.nodes[ id ][ attr_name ] = mappings[ graph.nodes[ id ][ attr_name ]]
    return class_labels.values()

def delete_graph_attributes( G ) :
    # delete all the graph attributes, since they are not useful for node classification

    # graph.graph['node_default'] = True
    # graph.graph['edge_default'] = True

    print( G.graph, type( G.graph ))

    if 'node_default' in G.graph :
        del G.graph[ 'node_default' ]
        
    if 'edge_default' in G.graph :
        del G.graph[ 'edge_default' ]

    # if 'edge_index' in G.graph:
    #     del G.graph['edge_index']

    attrs = [ attr for attr, _ in G.graph.items() ]

    for attr in attrs :
        del G.graph[ attr ]