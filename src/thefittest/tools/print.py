def print_net(net, ax=None):
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


def print_tree(some_tree, ax):
    import networkx as nx

    graph = some_tree.get_graph(False)

    G = nx.Graph()
    G.add_nodes_from(graph["nodes"])
    G.add_edges_from(graph["edges"])

    nx.draw_networkx_nodes(
        G, graph["pos"], node_color=graph["colors"], edgecolors="black", linewidths=0.5, ax=ax
    )
    nx.draw_networkx_edges(G, graph["pos"], style="-", ax=ax)
    nx.draw_networkx_labels(G, graph["pos"], graph["labels"], font_size=10, ax=ax)
