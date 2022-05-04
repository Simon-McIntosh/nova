"""Flow charts for Monte Carlo analysis of TFC vault assembly."""
from dataclasses import dataclass, field
from IPython.display import Image, display
from typing import ClassVar

import pydot


@dataclass
class Diagram:
    """Draw diagrams to illustrate Monte Carlo analyses."""

    theta: list[float] = field(default_factory=lambda: [1.5, 1.5, 2, 2, 5, 0])
    pdf: list[str] = field(
        default_factory=lambda: ['uniform', 'uniform', 'normal', 'normal',
                                 'uniform', 'uniform'])
    rankdir: str = 'UL'
    wall: bool = True

    samples: ClassVar[list[str]] = ['dr_case', 'dt_case', 'dr_ccl', 'dt_ccl',
                                    'dr_wall']

    def __post_init__(self):
        """Build graph."""
        self.graph = pydot.Dot('vault_monte_carlo', graph_type='digraph',
                               bgcolor='white', rankdir=self.rankdir)
        self.build_samples()
        self.build_models()
        self.build_nodes()
        self.build_links()

    def add_node(self, name, label, shape):
        """Add node to graph."""
        self.graph.add_node(pydot.Node(name, label=label, shape=shape))

    def add_edge(self, source, sink, **attrs):
        """Add edge to graph."""
        self.graph.add_edge(pydot.Edge(source, sink, **attrs))

    def sample_label(self, index):
        """Return sample label."""
        pdf = self.pdf[index]
        theta = self.theta[index]
        if pdf == 'normal':
            return f'N(0, {theta})'
        return f'U(+-{theta})'

    def build_samples(self, shape='box3d'):
        """Build graph nodes."""
        for i, name in enumerate(self.samples):
            if not self.wall and 'wall' in name:
                continue
            self.add_node(name, self.sample_label(i), shape)
        self.add_node('H', 'P(H)', shape)

    def build_models(self, shape='rectangle'):
        """Build model nodes."""
        self.add_node('build', 'Assembly', shape)
        self.add_node('ansys', 'Structural', shape)
        self.add_node('em', 'Electromagnetic', shape)
        if self.wall:
            self.add_node('wall', 'Wall Alignment', shape)
        self.add_node('lp', 'Lowpass (k<4)', shape)
        self.add_node('H99', 'H99', 'none')

    def build_nodes(self, shape='circle'):
        """Build summation nodes."""
        self.add_node('add_r', '+', shape)
        self.add_node('add_t', '+', shape)
        if self.wall:
            self.add_node('diffrence', '-', shape)

    def build_links(self):
        """Link nodes."""
        # sample input
        self.add_edge('dr_case', 'build', label=' dR case')
        self.add_edge('dt_case', 'build', label=' RdPhi case')

        # model links
        self.add_edge('build', 'ansys', label=' gap')
        self.add_edge('add_r', 'em', label=' dR em')
        self.add_edge('add_t', 'em', label=' RdPhi em')

        # summation nodes
        self.add_edge('ansys', 'add_r', label=' dR gap')
        self.add_edge('dr_case', 'add_r', label=' dR case')
        self.add_edge('dr_ccl', 'add_r', label=' dR ccl')

        self.add_edge('ansys', 'add_t', label=' RdPhi gap')
        self.add_edge('dt_case', 'add_t', label=' RdPhi case')
        self.add_edge('dt_ccl', 'add_t', label=' RdPhi ccl')

        if self.wall:
            self.add_edge('dr_wall', 'wall', label=' dR wall (machine)')
            self.add_edge('em', 'wall', label=' axis offset')
            self.add_edge('wall', 'diffrence', label=' dR wall (magnetic)')
            self.add_edge('em', 'diffrence', label=' fieldline')
            self.add_edge('diffrence', 'lp')
        else:
            self.add_edge('em', 'lp', label=' fieldline')

        self.add_edge('lp', 'H')
        self.add_edge('H', 'H99', label=' P(H<H99)=0.99')

    def plot(self):
        """Display graph."""
        display(Image(self.graph.create_png(), height=800))

    def write(self):
        """Save graph to file."""
        self.graph.write_png('tmp.png')



if __name__ == '__main__':

    diagram = Diagram([5, 5, 2, 2, 3, 0], wall=True)
    diagram.plot()
    diagram.write()
