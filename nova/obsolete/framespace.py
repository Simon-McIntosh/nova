# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:31:28 2021

@author: mcintos
"""

@dataclass
class Methods:
    """Manage FrameSpace methods."""

    frame: FrameSpace
    attrs: dict[Any] = field(repr=False, default_factory=dict)

    def __post_init__(self):
        """Define methods, update frame.columns and initialize methods."""
        self.frame.add_methods()
        self.initialize()

    def __repr__(self):
        """Return method list."""
        return f'{list(self.attrs)}'

    def initialize(self):
        """Init attrs derived from MetaMethod."""
        if self.frame.empty:
            return
        self.frame.update_columns()
        attrs = [attr for attr in self.attrs
                 if isinstance(self.attrs[attr], MetaMethod)]
        for attr in attrs:
            if self.attrs[attr].generate:
                self.attrs[attr].initialize()
