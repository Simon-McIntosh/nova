"""Manage frame index."""
from dataclasses import dataclass, field

import numpy as np

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame


@dataclass
class Label:
    """Manage selection labels."""

    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    preclude: list[str] = field(default_factory=lambda: [
        'feedback'], repr=False)
    require: list[str] = field(init=False)

    def __post_init__(self):
        """Type check input, extend exclude with preclude, ensure unique."""
        self.include = self.to_list(self.include)
        self.exclude = self.to_list(self.exclude)
        self.preclude = self.to_list(self.preclude)
        self.exclude.extend(
            [label for label in self.preclude if label not in self.exclude])
        for label in self.include:
            if label in self.exclude:
                self.exclude.remove(label)
        self.require = self.include + self.exclude

    def to_list(self, labels):
        """Convert None and str to lists, ([] and [str])."""
        if labels is None:
            return []
        if isinstance(labels, str):
            return [labels]
        return labels

    def to_dict(self):
        """Return label data as dict."""
        return {'include': self.include, 'exclude': self.exclude}


@dataclass
class Select(MetaMethod):
    """Manage dependant frame energization parameters."""

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: [
        'active', 'plasma', 'fix', 'feedback'], repr=False)
    require_all: bool = field(repr=False, default=False)
    additional: list[str] = field(init=False, default_factory=list, repr=False)
    avalible: list[str] = field(init=False, default_factory=list)
    labels: dict[str, dict[str, list[str]]] = field(init=False, repr=False,
                                                    default_factory=dict)

    def __post_init__(self):
        """Extend additional with unique values extracted from match."""
        if not self.generate:
            return
        self.add_label('active', 'active')
        self.add_label('passive', None, 'active')
        self.add_label('plasma', 'plasma')
        self.add_label('coil', None, ['plasma', 'passive'])
        self.add_label('fix', 'fix')
        self.add_label('free', None, 'fix')
        self.add_label('feedback', 'feedback')
        self.update_metaframe()  # update metaframe additional and defaults
        super().__post_init__()
        self.initialize()

    def __call__(self, required: list[str], require_all=False):
        """Update required attributes."""
        self.__init__(self.frame, required, require_all)

    def initialize(self):
        """Insert frame labels."""
        if not self.frame.empty and len(self.labels) > 0:
            self.update_columns()
            self.update_labels()

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
        additional = [label for label in list(dict.fromkeys(label.require))
                      if label not in self.additional]
        self.additional.extend(additional)

    def clear_labels(self):
        """Reset labels dict."""
        self.labels = {}

    def update(self):
        """Update frame with labels. Re-generate columns."""
        self.update_metaframe()
        super().__post_init__()  # add columns to avalible
        self.initialize()

    def update_metaframe(self):
        """
        Update metaframe.

        - Update additional.
        - Update defaults to include unset labels.
        """
        labels = list(self.labels)
        self.frame.metaframe.metadata = {
            'additional': labels, 'subspace': labels, 'array': labels}
        default = {label: True for label in self.labels
                   if label not in self.frame.metaframe.default}
        if len(default) > 0:
            self.frame.metaframe.metadata = {'default': default}

    def update_columns(self):
        """Update frame columns if any additional unset."""
        unset = np.array([label not in self.frame.columns
                          for label in self.additional])
        if unset.any():
            self.frame.update_columns()

    def update_labels(self):
        """Update frame selection labels."""
        for label in self.labels:
            include = self.any_label(self.labels[label]['include'], True)
            exclude = self.any_label(self.labels[label]['exclude'], False)
            with self.frame.setlock(True):
                self.frame.loc[:, label] = include & ~exclude

    def any_label(self, columns, default):
        """Return boolean index evaluated as columns.any()."""
        if columns:
            try:
                return self.frame.loc[:, columns].any(axis=1)
            except KeyError:
                pass
        return np.full(len(self.frame), default)


if __name__ == '__main__':

    dataframe = DataFrame({'x': range(4),
                           'plasma': [True, False, True, True]})
    select = Select(dataframe)
    #select.update()
    print(dataframe)
