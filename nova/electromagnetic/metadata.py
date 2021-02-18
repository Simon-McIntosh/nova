"""Manage CoilFrame metadata."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field, fields
import typing

# pylint:disable=unsubscriptable-object


@dataclass
class MetaData(metaclass=ABCMeta):
    """Abstract base class. Extended by MetaFrame and MetaArray."""

    update: dict[str, str] = field(default_factory=lambda: {})

    def __post_init__(self):
        """Validate input."""
        self.validate_input()

    @property
    def metadata(self):
        """Manage metadata."""
        return {field.name: getattr(self, field.name)
                for field in fields(self)}

    @metadata.setter
    def metadata(self, metadata):
        types = {field.name: typing.get_origin(field.type)
                 for field in fields(self)}
        [types[attr] not in [list, dict] for attr in ]
        for attr in [attr for attr in metadata if attr in types]:
            if not metadata[attr]:  # empty, None or False
                if types[attr] not in [list, dict]:

                value = [] if types[attr] == list else {}
                setattr(self, attr, metadata[attr])

            if types[attr] == list:


            else:  # replace if empty
                mode = 'replace'
            if mode == 'replace':

            elif mode == 'extend':
                unique = [attr for attr in metadata[attr]
                          if attr not in getattr(self, attr)]
                getattr(self, attr).extend(unique)
            elif mode == 'update':
                getattr(self, attr).update(metadata[attr])
            else:
                raise IndexError(f'mode {mode} not in '
                                 '[replace, extend, update]')
        self.validate_input()


    @abstractmethod
    def validate_input(self):
        """Run validation checks on input."""
        pass
