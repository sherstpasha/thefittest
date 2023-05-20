

def print_net(net, ax=None):
    import networkx as nx
    graph = net.get_graph()
    G = nx.DiGraph()
    G.add_nodes_from(graph['nodes'])
    G.add_edges_from(graph['connects'])

    nx.draw_networkx_nodes(G,
                           pos=graph['positions'],
                           node_color=graph['colors'],
                           edgecolors='black',
                           ax=ax)
    nx.draw_networkx_edges(G,
                           pos=graph['positions'],
                           edge_color=graph['weights_colors'],
                           ax=ax)

    nx.draw_networkx_labels(G,
                            graph['positions'],
                            graph['labels'],
                            ax=ax)
    
def print_tree(some_tree, ax):
    import networkx as nx
    graph = some_tree.get_graph(False)

    g = nx.Graph()
    g.add_nodes_from(graph['nodes'])
    g.add_edges_from(graph['edges'])

    nx.draw_networkx_nodes(g, graph['pos'], node_color=graph['colors'],
                           edgecolors='black', linewidths=0.5, ax=ax)
    nx.draw_networkx_edges(g, graph['pos'], ax=ax)
    nx.draw_networkx_labels(
        g, graph['pos'], graph['labels'], font_size=10, ax=ax)