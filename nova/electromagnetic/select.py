"""Manage frame index."""
from dataclasses import dataclass, field, fields

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame


@dataclass
class Label:
    """Manage selection labels."""

    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=lambda: ['feedback', 'plasma'])
    required: list[str] = field(init=False)

    def __post_init__(self):
        """Type check input."""
        for attr in ['include', 'exclude']:
            self.as_list(attr)
        for label in self.include:
            if label in self.exclude:
                self.exclude.remove(label)
        self.required = self.include + self.exclude

    def as_list(self, name):
        """Ensure all fields are lists."""
        attr = getattr(self, name)
        if not isinstance(attr, list):
            setattr(self, name, [attr])

    def to_dict(self):
        """Return label data as dict."""
        return {'include': self.include, 'exclude': self.exclude}


@dataclass
class Select(MetaMethod):
    """Manage dependant frame energization parameters."""

    frame: DataFrame = field(repr=False)
    required: list[str] = field(init=False, default_factory=lambda: [],
                                repr=False)
    require_all: bool = field(init=False, default=False, repr=False)
    additional: list[str] = field(init=False, default_factory=lambda: [],
                                  repr=False)
    avalible: list[str] = field(init=False, default_factory=lambda: [])
    labels: dict[str, dict[str, list[str]]] = field(
        init=False, default_factory=lambda: {}, repr=False)


    def __post_init__(self):
        """Extend additional with unique values extracted from match."""
        self.add_label('active', 'active')

        super().__post_init__()

    def add_label(self, name, *args):
        """
        Append selection label.

        Parameters
        ----------
        name : str
            Label name.
        include : list[str]
            Include flags.
        exclude : list[str]
            Exclude flags.

        Returns
        -------
        None.

        """
        label = Label(*args)
        self.labels[name] = label.to_dict()
        self.avalible.append(name)
        self.additional.extend([attr for attr in label.required
                                if label not in self.additional])

    '''
    def label_index(self):
        """Return label index."""
        if label == 'all' or label == 'full':  # all coils
            parts = self.frame.part

        elif label == 'free':  # optimize == True
            parts = coil.part[coil.optimize & ~coil.plasma & ~coil.feedback]
        elif label == 'fix':  # optimize == False
            parts = coil.part[~coil.optimize & ~coil.plasma & ~coil.feedback]
        else:
            if not pandas.api.types.is_list_like(label):
                label = [label]
            parts = self.coil.part
            parts = [_part for _part in label if _part in parts]
    '''


    def __call__(self, label):
        """Return label based selection."""
        return self.select(label)

    def initialize(self):
        """Metamethod initialization hook."""

    def select(self, label):
        """Return boolean selection index based on label."""
        if self.match[label] not in self.additional:
            raise IndexError(f'attr {self.match[label]} not specified '
                             f'in {self.additional}')
        index = getattr(self.frame, label)
        if label != self.match[label]:
            index = ~index  # negate complement
        for exclude in self.exclude:
            if label != exclude:
                index &= ~getattr(self.frame, exclude)
        return index


if __name__ == '__main__':

    dataframe = DataFrame({'x': range(3)},
                          Required=['x'], Additional=['Ic'],
                          Subspace=[], label='PF', Ic=3)
    select = Select(dataframe)

    print(select)
