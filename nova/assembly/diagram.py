"""Flow charts for Monte Carlo analysis of TFC vault assembly."""
from dataclasses import dataclass, field
from IPython.display import Image, display
from typing import ClassVar

import matplotlib
import pydot


@dataclass
class DiagramBase:
    """Diagram base class."""

    rankdir: str = 'UL'
    nodesep: float = 0.9
    ranksep: float = 0.1

    highlight: ClassVar[dict] = dict(penwidth=2, color='darkgreen')
    sample: ClassVar[dict] = dict(penwidth=2, color='darkorange')

    def __post_init__(self):
        """Build graph."""
        self.graph = pydot.Dot('vault_monte_carlo', graph_type='digraph',
                               bgcolor='white', rankdir=self.rankdir,
                               nodesep=self.nodesep, ranksep=self.ranksep)

    def add_node(self, name, label, shape, **kwargs):
        """Add node to graph."""
        self.graph.add_node(pydot.Node(name, label=label, shape=shape,
                                       **kwargs))

    def add_edge(self, source, sink, **attrs):
        """Add edge to graph."""
        self.graph.add_edge(pydot.Edge(source, sink, **attrs))

    def plot(self):
        """Display graph."""
        display(Image(self.graph.create('dot', 'png'), height=None))

    def write(self):
        """Save graph to file."""
        self.graph.write_png('tmp.png')


@dataclass
class Filter(DiagramBase):
    """Draw filter schematic."""

    rankdir: str = 'LR'
    nodesep: float = 2.5
    ranksep: float = 1.5

    def __post_init__(self):
        """Build model."""
        super().__post_init__()
        self.add_node('signal', 'signal', shape='rectangle',
                      color='darkblue', penwidth=2.5)
        self.add_node('filter', 'filter', shape='rectangle',
                      color='darkred', penwidth=2.5)
        self.add_node('response', 'response', shape='rectangle',
                      color='darkgreen', penwidth=2)
        self.add_edge('signal', 'filter', penwidth=2)
        self.add_edge('filter', 'response', penwidth=2)


@dataclass
class FFT(DiagramBase):
    """Draw ftt schematic."""

    rankdir: str = 'LR'
    nodesep: float = 1.5
    ranksep: float = 0.5

    def __post_init__(self):
        """Build model."""
        super().__post_init__()
        self.add_node('signal', 'signal', shape='rectangle',
                      color='darkblue', penwidth=2.5)
        self.add_node('fft', 'FFT', shape='rectangle',
                      color='k', penwidth=2.5)
        self.add_node('filter', 'Filter', shape='rectangle',
                      color='darkred', penwidth=2.5)
        self.add_node('ifft', 'inverse FFT', shape='rectangle',
                      color='k', penwidth=2.5)
        self.add_node('response', 'response', shape='rectangle',
                      color='darkgreen', penwidth=2)
        self.add_edge('signal', 'fft', penwidth=2)
        self.add_edge('fft', 'filter', label='Signal(k)', penwidth=2)
        self.add_edge('filter', 'ifft', label='Response(k)', penwidth=2)
        self.add_edge('ifft', 'response', penwidth=2)


@dataclass
class Overview(DiagramBase):
    """Draw Proxy models overview."""

    rankdir: str = 'UL'
    nodesep: float = 1.5
    ranksep: float = 0.75

    def __post_init__(self):
        """Build model."""
        super().__post_init__()
        self.build_samples()
        self.build_nodes()
        self.build_links()

    def build_samples(self, shape='box3d'):
        """Build graph nodes."""
        for sample in ['gap', 'roll', 'yaw']:
            self.add_node(sample, sample, shape, **self.sample)
        self.add_node('H', 'fieldline deviation', shape, **self.sample)

    def build_nodes(self, shape='rectangle'):
        """Build model nodes."""
        self.add_node(
            'ansys',
            'Structural\n----------------\n warm with gaps -> cold energized',
            shape, **self.highlight)
        self.add_node(
            'em',
            'Electromagnetic\n----------------\n '
            'structural displacement -> fieldline displacment',
            shape, **self.highlight)

    def build_links(self):
        """Link nodes."""
        # sample input
        for sample in ['gap', 'roll', 'yaw']:
            self.add_edge(sample, 'ansys', **self.highlight)
        self.add_edge('ansys', 'em', label=' radial',  **self.highlight)
        self.add_edge('ansys', 'em', label=' tangential',  **self.highlight)
        self.add_edge('em', 'H',  **self.highlight)


@dataclass
class Diagram(DiagramBase):
    """Draw diagrams to illustrate Monte Carlo analyses."""

    theta: list[float] = field(default_factory=lambda: [1.5, 1.5,
                                                        1.5, 1.5,
                                                        2, 2,
                                                        5])
    pdf: list[str] = field(
        default_factory=lambda: ['uniform', 'uniform',
                                 'uniform', 'uniform',
                                 'normal', 'normal',
                                 'uniform'])
    wall: bool = True

    samples: ClassVar[list[str]] = ['dr_case', 'dt_case',
                                    'roll', 'yaw',
                                    'dr_ccl', 'dt_ccl',
                                    'dr_wall']

    def __post_init__(self):
        """Build graph."""
        super().__post_init__()
        self.build_samples()
        self.build_models()
        self.build_nodes()
        self.build_links()

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
            self.add_node(name, self.sample_label(i), shape, **self.sample)
        self.add_node('H', 'P(H)', shape, **self.sample)

    def build_models(self, shape='rectangle'):
        """Build model nodes."""
        self.add_node('build', 'Assembly', shape)
        self.add_node('ansys', 'Structural', shape, **self.highlight)
        self.add_node('em', 'Electromagnetic', shape, **self.highlight)
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
        self.add_edge('build', 'ansys', label=' gap', **self.highlight)
        self.add_edge('roll', 'ansys', label=' roll', **self.highlight)
        self.add_edge('yaw', 'ansys', label=' yaw', **self.highlight)
        self.add_edge('add_r', 'em', label=' dR em', **self.highlight)
        self.add_edge('add_t', 'em', label=' RdPhi em', **self.highlight)

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


if __name__ == '__main__':

    # Diagram(theta=[5, 5, 5, 12, 2, 2, 2.6], wall=True).plot()
    # Overview().plot()
    # Filter().plot()
    FFT().plot()
