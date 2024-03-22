import numpy as np

from typing import Any
from typing import Tuple
from typing import List
from typing import Union

from ..base._net import Net
from ..base._net import NetEnsemble
from ..base import Tree


def print_net(net: Net, ax: Any = None) -> None:
    import networkx as nx

    graph = net.get_graph()
    G = nx.Graph()
    G.add_nodes_from(graph["nodes"])
    G.add_edges_from(graph["connects"])

    connects_label = {}
    for i, connects_i in enumerate(graph["connects"]):
        connects_label[tuple(connects_i)] = i

    nx.draw_networkx_nodes(
        G,
        pos=graph["positions"],
        node_color=graph["colors"],
        edgecolors="black",
        linewidths=0.5,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos=graph["positions"], style="-", edge_color=graph["weights_colors"], ax=ax, alpha=0.5
    )

    nx.draw_networkx_labels(G, graph["positions"], graph["labels"], font_size=10, ax=ax)


def print_tree(tree: Tree, ax: Any = None) -> None:
    import networkx as nx

    graph = tree.get_graph(True)

    G = nx.Graph()
    G.add_nodes_from(graph["nodes"])
    G.add_edges_from(graph["edges"])

    nx.draw_networkx_nodes(
        G, graph["pos"], node_color=graph["colors"], edgecolors="black", linewidths=0.5, ax=ax
    )
    nx.draw_networkx_edges(G, graph["pos"], style="-", ax=ax)
    nx.draw_networkx_labels(G, graph["pos"], graph["labels"], font_size=10, ax=ax)




def print_trees(trees: Union[List[Tree], Tuple[Tree]]) -> None:
    import matplotlib.pyplot as plt

    n_trees = len(trees)

    n_cols = max(2, int(np.ceil(np.sqrt(n_trees))))
    n_rows = max(2, int(np.ceil(n_trees / n_cols)))
    fig, ax = plt.subplots(figsize=(14, 7), ncols=n_cols, nrows=n_rows)

    counter: int = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if counter + 1 > n_trees:
                break

            print_tree(tree=trees[counter], ax=ax[i][j])
            counter += 1

    plt.tight_layout()


def print_nets(nets: Union[List[Net], Tuple[Net]]):
    import matplotlib.pyplot as plt
    n_nets = len(nets)

    n_cols = max(2, int(np.ceil(np.sqrt(n_nets))))
    n_rows = max(2, int(np.ceil(n_nets / n_cols)))
    fig, ax = plt.subplots(figsize=(14, 7), ncols=n_cols, nrows=n_rows)

    counter: int = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if counter + 1 > n_nets:
                break

            print_net(net=nets[counter], ax=ax[i][j])
            counter += 1

    plt.tight_layout()
    return fig

def print_ens(ens, ax=None):
    import networkx as nx
    nets = ens._nets
    meta_net = ens._meta_algorithm


    graph = nets[0].get_graph()

    print(graph)

    for net in nets[1:]:
        c = max(graph["nodes"]) + 1
        min_pos = min(v[1] for v in graph["positions"].values())

        graph_2 = net.get_graph()

        graph_2['nodes'] = [x + c for x in graph_2['nodes']]
        graph_2['labels'] = {k + c: v for k, v in graph_2['labels'].items()}
        graph_2['connects'] = graph_2['connects'] + c
        graph_2["positions"] = {k + c: np.array([v[0], v[1] + min_pos]) for k, v in graph_2["positions"].items()}

        graph["nodes"] = graph['nodes'] + graph_2['nodes']
        graph['labels'] = {**graph['labels'], **graph_2['labels']}
        graph['positions'] = {**graph['positions'], **graph_2['positions']}
        graph['colors'] = np.concatenate((graph['colors'], graph_2['colors']), axis=0)
        graph['weights_colors'] = np.concatenate((graph['weights_colors'], graph_2['weights_colors']), axis=0)
        graph['connects'] = np.concatenate((graph['connects'], graph_2['connects']), axis=0)

    
    G = nx.Graph()
    G.add_nodes_from(graph["nodes"])
    G.add_edges_from(graph["connects"])

    connects_label = {}
    for i, connects_i in enumerate(graph["connects"]):
        connects_label[tuple(connects_i)] = i

    nx.draw_networkx_nodes(
        G,
        pos=graph["positions"],
        node_color=graph["colors"],
        edgecolors="black",
        linewidths=0.5,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos=graph["positions"], style="-", edge_color=graph["weights_colors"], ax=ax, alpha=0.5
    )

    nx.draw_networkx_labels(G, graph["positions"], graph["labels"], font_size=10, ax=ax)




