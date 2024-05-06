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


def combine_and_adjust(list1, list2):
    # Смещение элементов второго списка до максимального значения равного 0
    max_val_list2 = max(list2)
    list2_shifted = [x - max_val_list2 for x in list2]

    # Дополнительное смещение, чтобы максимальное значение стало минимальным из первого списка минус один
    min_val_list1 = min(list1)
    list2_shifted = [x + (min_val_list1 - 1) for x in list2_shifted]

    # Комбинирование списков
    return list2_shifted


def print_ens(ens, ax=None):
    import networkx as nx

    nets = ens._nets
    meta_net = ens._meta_algorithm

    graph = nets[0].get_graph()
    outs = list(nets[0]._outputs)
    net_index = 0

    for k, v in graph["labels"].items():
        if k in nets[0]._outputs:
            graph["labels"][k] = str(v) + f"_{net_index}"
    net_index = net_index + 1

    for net in nets[1:]:
        c = max(graph["nodes"]) + 1

        graph_2 = net.get_graph()
        for k, v in graph_2["labels"].items():
            if k in net._outputs:
                graph_2["labels"][k] = str(v) + f"_{net_index}"

        net_index = net_index + 1

        outs2 = list(net._outputs)

        graph_v1 = [v[1] for k, v in graph["positions"].items()]
        graph_2_v1 = [v[1] for k, v in graph_2["positions"].items()]
        graph_2_v1_shifted = combine_and_adjust(graph_v1, graph_2_v1)

        graph_2["nodes"] = [x + c for x in graph_2["nodes"]]
        graph_2["labels"] = {k + c: v for k, v in graph_2["labels"].items()}
        graph_2["connects"] = graph_2["connects"] + c

        graph_2["positions"] = {
            k + c: np.array([v[0], v_new])
            for (k, v), v_new in zip(graph_2["positions"].items(), graph_2_v1_shifted)
        }
        outs2 = [x + c for x in outs2]

        graph["nodes"] = graph["nodes"] + graph_2["nodes"]
        graph["labels"] = {**graph["labels"], **graph_2["labels"]}
        graph["positions"] = {**graph["positions"], **graph_2["positions"]}
        graph["colors"] = np.concatenate((graph["colors"], graph_2["colors"]), axis=0)
        graph["weights_colors"] = np.concatenate(
            (graph["weights_colors"], graph_2["weights_colors"]), axis=0
        )
        graph["connects"] = np.concatenate((graph["connects"], graph_2["connects"]), axis=0)
        outs = outs + outs2

    x_poses = [v[0] for k, v in graph["positions"].items() if k in outs]

    max_poses = max(x_poses)

    for k, v in graph["positions"].items():
        if k in outs:
            v[0] = max_poses

    graph_meta = meta_net.get_graph()
    c = max(graph["nodes"]) + 1
    outs.append(max(outs) + 1)

    mapping = {list(meta_net._inputs)[i]: outs[i] for i in range(len(outs))}
    new_nodes = [mapping.get(item, item + c) for item in graph_meta["nodes"]]
    new_labels = {mapping.get(key, key + c): value for key, value in graph_meta["labels"].items()}

    new_connects = graph_meta["connects"].copy()
    new_positions = {
        mapping.get(k, k + c): np.array([v[0] + max_poses + 1, v[1]])
        for (k, v) in graph_meta["positions"].items()
    }

    for col in range(new_connects.shape[1]):  # Для каждого столбца
        for i, val in enumerate(list(meta_net._inputs)):
            new_connects[new_connects[:, col] == val, col] = outs[i]

        mask = ~np.isin(new_connects[:, col], outs)
        new_connects[:, col][mask] += c

    print("---")
    print(len(new_nodes))
    print(len(new_labels))
    print(len(new_connects))
    print(len(new_positions))
    print(len(graph_meta["colors"]))
    print(len(graph_meta["weights_colors"]))

    # print(graph["labels"])
    # print(new_labels)

    # print(graph["nodes"])
    # print(new_nodes)

    # print(list(set((graph["nodes"] + new_nodes))))

    print(graph["positions"])
    print(new_positions)

    # for key, value in new_labels:
    #     if key not in graph["labels"]:
    #         graph["labels"][key] = value

    # for item in new_labels:
    #     if item not in graph["nodes"]:
    #         graph["nodes"].append(item)

    # graph["nodes"] = list(set((graph["nodes"] + new_nodes)))
    # graph["labels"] = {**graph["labels"], **new_labels}
    # graph["positions"] = {**graph["positions"], **new_positions}
    # graph["colors"] = np.concatenate(
    #     (graph["colors"], graph_meta["colors"][len(meta_net._inputs) - 1 :]), axis=0
    # )
    # graph["weights_colors"] = np.concatenate(
    #     (graph["weights_colors"], graph_meta["weights_colors"]), axis=0
    # )
    # graph["connects"] = np.concatenate((graph["connects"], new_connects), axis=0)

    print("---")
    print(len(graph["nodes"]))
    print(len(graph["labels"]))
    print(len(graph["positions"]))
    print(len(graph["colors"]))
    print(len(graph["weights_colors"]))
    print(len(graph["connects"]))

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
