"""Manage CoilFrame metadata."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields
import typing

# pylint:disable=unsubscriptable-object


@dataclass
class MetaData(metaclass=ABCMeta):
    """Abstract base class. Extended by MetaFrame and MetaArray."""

    def __post_init__(self):
        """Validate input."""
        self.validate()

    @property
    def types(self) -> dict[str, type]:
        """Return field types."""
        return {field.name: typing.get_origin(field.type)
                if isinstance(field.type, typing.GenericAlias) else field.type
                for field in fields(self)}

    @property
    def metadata(self):
        """
        Manage metadata.

        Parameters
        ----------
        metadata : dict[str, Union[list, dict]]
            Input metadata.
                - if not value (empty, False, None): clear attribute
                - if attribute[0].isupper(): replace field with value
                - else: update / extend

        Returns
        -------
        metadata : dict[str, Union[list, dict]]

        """
        return {field.name: getattr(self, field.name)
                for field in fields(self)}

    @metadata.setter
    def metadata(self, metadata):
        types = self.types
        for attribute in [attr for attr in metadata if attr.lower() in types]:
            replace = attribute[0].isupper()
            value = metadata[attribute]
            attribute = attribute.lower()
            if not value:  # empty, None or False
                setattr(self, attribute, types[attribute]())
            else:
                if not isinstance(value, types[attribute]):
                    raise TypeError('type missmatch: '
                                    'type(input) != type(default) \n'
                                    f'{type(metadata[attribute])} != '
                                    f'{types[attribute]}')
                if replace:
                    setattr(self, attribute, value)
                elif types[attribute] == dict:
                    getattr(self, attribute).update(value)
                elif types[attribute] == list:
                    getattr(self, attribute).extend(
                        [attr for attr in value
                         if attr not in getattr(self, attribute)])
                else:
                    raise TypeError(f'attribute type {types[attribute]} ',
                                    'not in [list, dict]')
        self.validate()

    @abstractmethod
    def validate(self):
        """Run validation checks on input."""
        types = self.types
        type_error = {name: types[name] for name in types
                      if types[name] not in [list, dict]}
        if type_error:
            raise TypeError('attributes initialized with types '
                            'not in [list, dict]:\n'
                            f'{type_error}')

    def clear(self, attribute):
        """
        Replace named attribute with empty strucutre matching specified type.

        Parameters
        ----------
        attribute : str
            Atribute to clear.

        Raises
        ------
        AttributeError
            Attribute not found.

        Returns
        -------
        None.

        """
        if attribute not in self.types:
            raise AttributeError(f'attribute {attribute} not found')
        self.metadata = {attribute: self.types[attribute]()}
