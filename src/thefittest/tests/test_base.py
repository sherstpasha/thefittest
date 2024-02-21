# import numpy as np

# from ..base._ea import Statistics
# from ..base._ea import TheFittest
# from ..base._net import HiddenBlock
# from ..base._net import Net
# from ..base._tree import Add
# from ..base._tree import EphemeralConstantNode
# from ..base._tree import EphemeralNode
# from ..base._tree import FunctionalNode
# from ..base._tree import Mul
# from ..base._tree import Operator
# from ..base._tree import Sub
# from ..base._tree import TerminalNode
# from ..base._tree import Tree
# from ..base._tree import UniversalSet


# class PlusOne(Operator):
#     def __init__(self) -> None:
#         Operator.__init__(self, formula="({} + 1)", name="plus_one", sign="+1")

#     def __call__(self, x: int) -> int:
#         result = x + 1
#         return result


# def test_thefittest_class():
#     net_1 = Net()
#     net_2 = Net()
#     net_3 = Net()

#     population_g = np.array([net_1, net_2, net_3], dtype=object)
#     population_ph = population_g
#     fitness = np.array([1, 2, 3], dtype=np.float64)

#     thefittest_ = TheFittest()
#     thefittest_._update(population_g, population_ph, fitness)
#     assert thefittest_._fitness == 3
#     assert thefittest_._no_update_counter == 0
#     assert thefittest_._fitness is not fitness[2]
#     assert thefittest_._genotype is not population_g[2]
#     assert thefittest_._phenotype is not population_ph[2]

#     fitness = np.array([1, 2, 2], dtype=np.float64)
#     thefittest_._update(population_g, population_ph, fitness)
#     assert thefittest_._no_update_counter == 1
#     assert thefittest_._fitness == 3

#     fitness = np.array([1, 2, 4], dtype=np.float64)
#     thefittest_._update(population_g, population_ph, fitness)
#     assert thefittest_._no_update_counter == 0
#     assert thefittest_._fitness == 4

#     fittest = thefittest_.get()
#     assert isinstance(fittest, dict)
#     assert len(fittest) == 3


# def test_statistic_class():
#     net_1 = Net()
#     net_2 = Net()
#     net_3 = Net()
#     population_g = np.array([net_1, net_2, net_3], dtype=object)
#     population_ph = population_g
#     fitness = np.array([1, 2, 3], dtype=np.float64)

#     statistics_ = Statistics()
#     fitness_max = np.max(fitness)
#     statistics_._update(
#         {"population_g": population_g, "population_ph": population_ph, "max_fitness": fitness_max}
#     )

#     assert statistics_["max_fitness"][0] == fitness_max
#     assert np.all(statistics_["population_g"][0] == population_g)
#     assert np.all(statistics_["population_ph"][0] == population_ph)

#     statistics_._update(
#         {"population_g": population_g, "population_ph": population_ph, "max_fitness": fitness_max}
#     )

#     assert len(statistics_["max_fitness"]) == 2
#     assert len(statistics_["population_g"]) == 2
#     assert len(statistics_["population_ph"]) == 2


# def test_nodes_class():
#     def generator():
#         return 10

#     functional_node = FunctionalNode(value=Add(), sign="+")
#     assert functional_node._n_args == 2
#     assert functional_node._sign == "+"
#     assert functional_node._value(1, 2) == 3

#     functional_node2 = FunctionalNode(value=PlusOne())
#     assert functional_node2._sign == "+1"
#     assert functional_node2._value(1) == 2

#     functional_node3 = FunctionalNode(value=Add())
#     assert functional_node3._sign == Add()._sign

#     terminal_node = TerminalNode(value=1, name="1")
#     assert str(terminal_node) == "1"

#     terminal_node2 = TerminalNode(value=2, name="2")

#     assert terminal_node != terminal_node2

#     ephemeral_node = EphemeralNode(generator)
#     ephemeral_constant_node = ephemeral_node()
#     assert isinstance(ephemeral_constant_node, EphemeralConstantNode)

#     universal_set = UniversalSet(
#         functional_set=(functional_node, functional_node2, functional_node3),
#         terminal_set=(terminal_node, terminal_node2),
#     )

#     random_terminal = universal_set._random_terminal_or_ephemeral()
#     assert random_terminal in universal_set._terminal_set

#     random_functional_2 = universal_set._random_functional(n_args=2)
#     assert random_functional_2 in universal_set._functional_set[2]
#     assert random_functional_2._n_args == 2

#     random_functional_1 = universal_set._random_functional(n_args=1)
#     assert random_functional_1 in universal_set._functional_set[1]
#     assert random_functional_1._n_args == 1
#     assert random_functional_1 is functional_node2

#     universal_set = UniversalSet(
#         functional_set=(functional_node, functional_node2, functional_node3),
#         terminal_set=(ephemeral_node,),
#     )
#     random_ephemeral = universal_set._random_terminal_or_ephemeral()
#     assert isinstance(random_ephemeral, EphemeralConstantNode)
#     assert random_ephemeral._value == 10


# def test_tree_class():
#     x = TerminalNode(value=5, name="x")
#     y = TerminalNode(value=7, name="y")
#     z = TerminalNode(value=11, name="z")

#     functional_add = FunctionalNode(value=Add(), sign="+")
#     functional_mul = FunctionalNode(value=Mul(), sign="*")
#     functional_sub = FunctionalNode(value=Sub(), sign="-")

#     tree = Tree(nodes=[functional_mul, z, functional_add, x, y])

#     tree2 = Tree(
#         nodes=[functional_add, functional_mul, x, x, functional_mul, z, functional_sub, y, x]
#     )

#     assert tree() == 132
#     assert str(tree) == "(z * (x + y))"
#     assert len(tree) == 5

#     assert tree2() == 47
#     assert str(tree2) == "((x * x) + (z * (y - x)))"
#     assert len(tree2) == 9

#     assert tree2 == tree2
#     assert bool(tree2 == tree) is False

#     tree_copy = tree.copy()
#     assert tree_copy == tree
#     assert tree_copy is not tree
#     assert tree_copy._nodes is not tree._nodes
#     assert tree_copy._n_args is not tree._n_args

#     tree3 = tree2.set_terminals(x=3, y=4, z=5)
#     assert tree3() == 14
#     assert bool(tree2 == tree3) is True

#     tree3._nodes[0] = functional_mul
#     assert bool(tree2 == tree3) is False

#     index, n_index = tree2.subtree_id(4)
#     assert (index, n_index) == (4, 9)

#     tree4 = tree2.subtree(4)
#     assert tree4() == 22
#     assert str(tree4) == "(z * (y - x))"
#     assert len(tree) == 5

#     tree5 = tree.concat(index=4, other_tree=tree)
#     assert str(tree5) == "(z * (x + (z * (x + y))))"
#     assert tree5() == 1507

#     args_id = tree.get_args_id(index=2)
#     assert np.all(args_id == np.array([3, 4], dtype=np.int64))

#     levels = tree.get_levels(index=0)
#     assert np.all(levels == np.array([0, 1, 1, 2, 2], dtype=np.int64))

#     max_level = tree.get_max_level()
#     assert max_level == 2

#     graph = tree.get_graph()
#     graph = tree.get_graph(keep_id=True)

#     assert isinstance(graph, dict)


# def test_net():
#     net = Net(inputs={0, 1, 3})
#     net2 = Net(inputs={2})

#     net3 = net + net2
#     assert net3._inputs == {0, 1, 2, 3}

#     net4 = net > net2
#     assert net4._inputs == {0, 1, 2, 3}

#     net5 = Net(hidden_layers=[{4}])
#     net6 = Net(hidden_layers=[{5, 6}])

#     net7 = net5 + net6

#     assert net7._hidden_layers == [{4, 5, 6}]

#     net8 = net5 > net6
#     assert net8._hidden_layers == [{4}, {5, 6}]
#     assert np.all(net8._connects == np.array([[4, 5], [4, 6]], dtype=np.int64))

#     net9 = net2 + net5
#     assert np.all(net9._connects == np.array([[2, 4]], dtype=np.int64))

#     net10 = net5 + net2
#     assert np.all(net10._connects == np.array([[2, 4]], dtype=np.int64))

#     net11 = net5 > net2
#     assert np.all(net11._connects == np.array([[2, 4]], dtype=np.int64))

#     net12 = Net(hidden_layers=[{4}, {5, 6}])
#     net13 = Net(hidden_layers=[{7, 8}])

#     net14 = net12 + net13
#     assert net14._hidden_layers == [{4, 7, 8}, {5, 6}]

#     net15 = net13 + net12
#     assert net15._hidden_layers == [{4, 7, 8}, {5, 6}]

#     connects, weights = net._get_connect({1, 2}, {})
#     assert np.all(connects == np.zeros((0, 2), dtype=int))
#     assert np.all(weights == np.zeros((0), dtype=float))

#     assert len(net11) == 1

#     fixed = net5._fix(inputs={1, 2})

#     assert np.all(fixed._connects == np.array([[1, 4], [2, 4]], dtype=np.int64))
#     assert fixed._inputs == {1, 2}

#     net16 = Net(
#         inputs={0, 1},
#         hidden_layers=[{2, 3}, {4}],
#         outputs={5},
#         connects=np.array([[0, 2], [1, 3], [2, 4], [3, 4], [4, 5]], dtype=np.int64),
#         weights=np.array([0.1, 0.2, 0.4, 0.5, 1.5], dtype=np.float64),
#         activs={2: 1, 3: 1, 4: 1, 5: 1},
#     )

#     X = np.array([[1, 2]], dtype=np.float64)
#     out = net16.forward(X)

#     print(out)

#     graph = net16.get_graph()

#     assert out[0][0][0] == 0.36000000000000004
#     assert isinstance(graph, dict)

#     hb = HiddenBlock(max_size=10)

#     assert hb._activ in [0, 1, 2, 3, 4]
#     assert hb._size <= 10

#     net17 = net16.copy()
#     net17._inputs.add(3)

#     net18 = net16.copy()
#     net18._outputs.add(9)

#     net19 = net16.copy()
#     net19._activs[2] = 2

#     net20 = net16.copy()
#     net20._hidden_layers[0].add(22)

#     net21 = net16.copy()
#     net21._connects[0][1] = 3

#     assert net16 == net16
#     for i, net_i in enumerate([net16, net17, net18, net19, net20, net21]):
#         for j, net_j in enumerate([net16, net17, net18, net19, net20, net21]):
#             if i == j:
#                 assert net_i == net_j
#             else:
#                 assert net_i != net_j
