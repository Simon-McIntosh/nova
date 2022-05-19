import numpy as np
import networkx as nx

matrix = np.array([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]])

incidence_matrix = -matrix[:, ::2] + matrix[:, 1::2]
edge_list = [(np.where(col == -1)[0][0], np.where(col == 1)[0][0])
             for col in incidence_matrix.T]
G = nx.Graph(edge_list)
DiG = nx.DiGraph(edge_list)
pos = nx.spring_layout(G, seed=2025)

pos = {0: (0, 1), 1: (-1, 0), 2: (0, 0), 3: (1, 0), 4: (0, -1)}
nx.draw(DiG, pos, with_labels=True)

print(list(nx.cycle_basis(G)))

for loop in nx.cycle_basis(G):
    loop.append(loop[0])
    pairs = [tuple(loop[i:i+2]) for i in range(len(loop) - 1)]
    edges = []
    for pair in pairs:
        try:
            edges.append(edge_list.index(pair))
        except ValueError:
            edges.append(-edge_list.index(pair[::-1]))
    print(edges)
