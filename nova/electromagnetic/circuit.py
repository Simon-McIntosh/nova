"""Manage coilset supplies."""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import xarray

from nova.database.netcdf import netCDF
from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.utilities.pyplot import plt


@dataclass
class Circuit(netCDF, FrameSetLoc):
    """Manage coil supplies."""

    name: str = 'circuit'

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
            np.zeros((self.data.dims['node'], self.data.dims['edge']))
        data['incidence_matrix'][:len(connection)] = \
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
        dig = nx.DiGraph(edge_list.values())
        pos = nx.planar_layout(dig)

        edge_labels = {edge_list[edge]: edge for edge in edge_list}
        if len(edge_list) == 2:
            nx.draw_networkx_edges(dig, pos, connectionstyle='arc3,rad=0.15')
            nx.draw_networkx_nodes(dig, pos)
            nx.draw_networkx_labels(dig, pos)
            nx.draw_networkx_edge_labels(dig, pos, edge_labels=edge_labels,
                                         label_pos=0.3)
            plt.gca().set_axis_off()
            return
        nx.draw(dig, pos, with_labels=True)
        nx.draw_networkx_edge_labels(dig, pos, edge_labels=edge_labels,
                                     label_pos=0.5)

    def plot_all(self):
        """Plot all circuits."""
        for circuit in self.data.circuit:
            plt.figure()
            self.plot(circuit)

    def edge_loops(self, circuit: str):
        """Extract basis loops."""
        edge_list = self.edge_list(circuit)
        edge_nodes = edge_list.values()
        graph = nx.Graph(edge_list)
        loops = []
        for loop in nx.cycle_basis(graph):
            edge = [_loop for _loop in loop if isinstance(_loop, str)]
            nodes = [_loop for _loop in loop if not isinstance(_loop, str)]
            nodes.append(nodes[0])
            pairs = [tuple(nodes[i:i+2]) for i in range(len(nodes) - 1)]
            sign = np.ones(len(pairs), dtype=int)
            for i, pair in enumerate(pairs):
                if pair in edge_nodes:
                    continue
                if pair[::-1] in edge_nodes:
                    sign[i] = -1
                    continue
                raise IndexError(f'node pair {pair} '
                                 f'not in edge nodes {edge_nodes}')
            for i, name in enumerate(edge):
                if name in self.frame.index:
                    continue
                if sign[i] == -1:
                    sign *= -1
            loops.append(dict(edge=edge, sign=sign))
        return loops

    def link(self):
        """Link single circuit coils."""
        for circuit in self.data.circuit.values:
            loops = self.edge_loops(circuit)
            if len(loops) > 1:
                continue
            loop = loops[0]
            index, factor = [], []
            for edge, sign in zip(loop['edge'], loop['sign']):
                if edge in self.frame.index:
                    index.append(edge)
                    factor.append(sign)
            if len(index) == 1:
                continue
            if factor[0] == -1:
                factor = [-f for f in factor]
            sort = sorted(zip(index, factor),
                          key=lambda x: self.frame.index.get_loc(x[0]))
            index, factor = list(map(list, zip(*sort)))
            self.linkframe(index, factor[1:])
