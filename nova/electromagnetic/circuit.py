"""Manage coilset supplies."""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import xarray

from nova.database.netcdf import netCDF
from nova.electromagnetic.framesetloc import FrameSetLoc


@dataclass
class Circuit(netCDF, FrameSetLoc):
    """Manage coil supplies."""

    def initialize(self, supply: list[str], nodes: int):
        """Initialize dataset."""
        self.data = xarray.Dataset()
        self.data['supply'] = supply
        self.data['coil'] = self.sloc().index.values
        self.data['edge'] = np.append(supply, self.sloc().index.values)
        self.data['node'] = np.arange(nodes)
        self.data['circuit'] = np.array([], str)

    def insert(self, circuit: str, connection):
        """Insert unsigned directed incidence matrix."""
        data = self.data.coords.to_dataset()
        data['circuit'] = [circuit]
        connection = connection[:, :2*self.data.dims['edge']]
        data['incidence_matrix'] = ('node', 'edge'), \
            -connection[:, ::2] + connection[:, 1::2]
        self.data = xarray.concat([self.data, data], 'circuit')

    def edge_list(self, circuit: str):
        """Return circuit edge list."""
        incidence_matrix = self.data['incidence_matrix'].sel(circuit=circuit)
        return {edge: (np.where(col == -1)[0][0], np.where(col == 1)[0][0])
                for edge, col in zip(self.data.edge.values,
                                     incidence_matrix.values.T)
                if (-1 in col and 1 in col)}

    def plot(self, circuit: str):
        """Plot directed graph."""
        edge_list = self.edge_list(circuit)
        DiG = nx.DiGraph(edge_list.values())
        pos = nx.planar_layout(DiG)

        nx.draw(DiG, pos, with_labels=True)
        edge_labels = {edge_list[edge]: edge for edge in edge_list}
        nx.draw_networkx_edge_labels(DiG, pos, edge_labels=edge_labels)
        nx.draw_networkx_nodes(DiG, pos)
        #nx.draw_networkx_edges(DiG, pos,
        #                       connectionstyle='arc3,rad=0.2')

    def edge_loops(self, circuit: str):
        """Extract basis loops."""
        edge_list = self.edge_list(circuit)
        G = nx.Graph(edge_list)
        for loop in nx.cycle_basis(G):
            loop.append(loop[0])
            pairs = [tuple(loop[i:i+2]) for i in range(len(loop) - 1)]
            edge_index = np.zeros(len(pairs), dtype=int)
            direction = np.ones(len(pairs), dtype=int)
            for i, pair in enumerate(pairs):
                try:
                    edge_index[i] = edge_list.index(pair)
                except ValueError:
                    edge_index[i] = edge_list.index(pair[::-1])
                    direction[i] = -1
            #[for index in edge_index]
            #print(edge_list[edge], direction)
            #print(self.data.edge[edge].values)



'''

import numpy as np

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

'''
