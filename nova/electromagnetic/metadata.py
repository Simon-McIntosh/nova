"""Manage CoilFrame metadata."""
from abc import ABC
from dataclasses import dataclass, field, fields

import typing

# pylint:disable=unsubscriptable-object


@dataclass
class MetaData(ABC):
    """Abstract base class. Extended by MetaFrame."""

    # internal field list - exclude from metadata return
    _internal: list[str] = field(init=False, default_factory=list)

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
                for field in fields(self) if field.name not in self._internal}

    @metadata.setter
    def metadata(self, metadata):
        if metadata is None:
            metadata = {}
        types = self.types
        for attribute in [attr for attr in metadata if attr.lower() in types]:
            replace = attribute[0].isupper()
            value = metadata[attribute]
            attribute = attribute.lower()
            if not value:  # [], None or False
                setattr(self, attribute, types[attribute]())
            else:
                if not isinstance(value, types[attribute]):
                    if isinstance(value, str):
                        value = [value]
                    else:
                        raise TypeError('type missmatch: '
                                        'type(input) != type(default) \n'
                                        f'attribute: {attribute}\n'
                                        f'value: {value}\n'
                                        f'{type(metadata[attribute])} != '
                                        f'{types[attribute]}')
                if replace:
                    setattr(self, attribute, value)
                elif types[attribute] == dict:
                    getattr(self, attribute).update(value)
                elif types[attribute] == list:
                    getattr(self, attribute).extend(
                        [attr for attr in list(dict.fromkeys(value))
                         if attr not in getattr(self, attribute)])
                else:
                    raise TypeError(f'attribute type {types[attribute]} ',
                                    'not in [list, dict]')
        self.validate()

    def validate(self):
        """Run validation checks on input."""
        types = self.types
        type_error = {field.name: [types[field.name],
                                   type(getattr(self, field.name))]
                      for field in fields(self)
                      if not isinstance(getattr(self, field.name),
                                        types[field.name])}
        if type_error:
            raise TypeError('attributes initialized with incorrect type:\n'
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
