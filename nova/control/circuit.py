"""Manage coilset supplies."""
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module

import numpy as np
import xarray

from nova.database.netcdf import netCDF
from nova.frame.baseplot import Plot
from nova.frame.framesetloc import FrameSetLoc
from nova.frame.framespace import FrameSpace


@dataclass
class Circuit(Plot, netCDF, FrameSetLoc):
    """Manage coil supplies."""

    name: str = 'circuit'
    supply: FrameSpace = field(default_factory=FrameSpace)
    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)

    def __add__(self, other):
        """Return union of self and other."""
        supply = self.supply + other.supply
        data = self.data.merge(other.data, combine_attrs='drop_conflicts')
        return Circuit(*self.frames, supply=supply, data=data)

    def __iadd__(self, other):
        """Return self with data augmented by other."""
        self.supply += other.supply
        self.data = self.data.merge(other.data, combine_attrs='drop_conflicts')
        return self

    def initialize(self, supply: list[str], nodes: int):
        """Initialize dataset."""
        self.data = xarray.Dataset()
        self.data['supply'] = supply
        self.data['coil'] = self.Loc().index.values
        self.data['edge'] = np.append(supply, self.Loc().index.values)
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

    def plot(self, circuit: str, axes=None):
        """Plot directed graph."""
        self.set_axes('2d', axes=axes)
        edge_list = self.edge_list(circuit)
        networkx = import_module('networkx')
        dig = networkx.DiGraph(edge_list.values())
        pos = networkx.planar_layout(dig)
        edge_labels = {edge_list[edge]: edge for edge in edge_list}
        if len(edge_list) == 2:
            networkx.draw_networkx_edges(dig, pos,
                                         connectionstyle='arc3,rad=0.15')
            networkx.draw_networkx_nodes(dig, pos)
            networkx.draw_networkx_labels(dig, pos)
            networkx.draw_networkx_edge_labels(
                dig, pos, edge_labels=edge_labels, label_pos=0.3)
            return
        networkx.draw(dig, pos, with_labels=True)
        networkx.draw_networkx_edge_labels(
            dig, pos, edge_labels=edge_labels, label_pos=0.5)

    def plot_all(self):
        """Plot all circuits."""
        for circuit in self.data.circuit:
            self.plot(circuit, None)

    def edge_loops(self, circuit: str):
        """Return basis loops."""
        edge_list = self.edge_list(circuit)
        edge_nodes = edge_list.values()
        networkx = import_module('networkx')
        graph = networkx.Graph(edge_list)
        loops = []
        for loop in networkx.cycle_basis(graph):
            edge = [_loop for _loop in loop if isinstance(_loop, str)]
            nodes = [_loop for _loop in loop if not isinstance(_loop, str)]
            nodes = [nodes[-1]] + nodes
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
            loops.append(dict(edge=edge, sign=sign))
        return loops

    @cached_property
    def loops(self):
        """Return list of all circuit loops."""
        return [loop for circuit in self.data.circuit.data
                for loop in self.edge_loops(circuit)]

    @cached_property
    def loop_number(self):
        """Return total loop number."""
        return len(self.loops)

    def _extract_linked_coils(self, loop):
        """Extract coil edge and orientations from loop."""
        index = np.array([np.where(coil == np.array(loop['edge']))[0][0]
                          for coil in self.data.coil.data
                          if coil in np.array(loop['edge'])], dtype=int)
        if len(index) < 2:
            return None
        return {attr: list(np.array(loop[attr])[index])
                for attr in ['edge', 'sign']}

    @cached_property
    def links(self):
        """Return list of all single circuits with more than one coil."""
        return [loop for circuit in self.data.circuit.data
                if len(loops := self.edge_loops(circuit)) == 1 and
                (loop := self._extract_linked_coils(loops[0])) is not None]

    @cached_property
    def link_number(self):
        """Return single circuit coil link number."""
        return np.sum([len(link['edge'])-1 for link in self.links])

    def _coupling_matrix(self, column: str):
        """Return loop conectivity matrix."""
        vector = self.data[column].data
        matrix = np.zeros((self.loop_number + self.link_number,
                           self.data.dims[column]), float)
        for i, loop in enumerate(self.loops):
            matrix[i] = np.sum([
                np.where(vector == edge, sign, 0)
                for edge, sign in zip(loop['edge'], loop['sign'])], axis=0)
        return matrix

    def supply_matrix(self):
        """Return supply conectivity matrix."""
        return self._coupling_matrix('supply')

    def coil_matrix(self):
        """Return coil conectivity matrix."""
        return self._coupling_matrix('coil')

    def link_matrix(self):
        """Return single loop link matrix."""
        coils = self.data.coil.data
        matrix = np.zeros((self.link_number,
                           self.data.dims['coil']), float)
        index = 0
        for link in self.links:
            reference = np.where(coils == link['edge'][0], link['sign'][0], 0)
            for i in range(len(link['edge']) - 1):
                matrix[index] = np.sum([reference,
                                        np.where(coils == link['edge'][i+1],
                                                 -link['sign'][i+1], 0)],
                                       axis=0)
                index += 1
        return matrix

    def link(self):
        """Link single circuit coils."""
        for circuit in self.data.circuit.data:
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
