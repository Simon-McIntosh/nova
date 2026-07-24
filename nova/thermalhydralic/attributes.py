"""Manage data attributes."""

import pandas
import numpy as np


class Attributes:
    """
    Manage data attributes.

    Parameters
    ----------
    _attributes : array-like
        List of attribute names.
    _default_attributes : dict
        Dict of default attribute / value pairs.
    _input_attributes : array-like
        Minimal list of input attributes required for subsiquent tasks to run.

    """

    def __init__(self):
        self._attributes = []
        self._default_attributes = {}
        self._input_attributes = []
        self.attributes = ["dummy1", "dummy2"]

    @property
    def attributes(self):
        """
        Manage data attributes.

        Parameters
        ----------
        attributes : str or array-like
            Additional attributes.
            Appended to self._attributes if not already present.

        Returns
        -------
        attributes : array-like
            List of data attributes.

        """
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        if not pandas.api.types.is_list_like(attributes):
            attributes = [attributes]
        self._attributes.extend(
            [attr for attr in attributes if attr not in self._attributes]
        )

    @property
    def default_attributes(self):
        """
        Manage default attributes.

        Parameters
        ----------
        default_attributes : dict
            Dict of default attributes, updates self._default_attriutes.

        Returns
        -------
        default_attributes : dict

        """
        return self._default_attributes

    @default_attributes.setter
    def default_attributes(self, default_attributes):
        self._default_attributes.update(default_attributes)

    def initialize_attributes(self):
        """Initialize data attributes."""
        for attribute in self.attributes:
            setattr(self, f"_{attribute}", None)
        for attribute in self.default_attributes:
            setattr(self, f"_{attribute}", self._default_attributes[attribute])

    def set_attributes(self, *args, **kwargs):
        """
        Set data attributes via args or kwargs.

        Order of evaluation:
            - args
            - kwargs
            - namespace variable that is not None
            - _default_attributes

        Parameters
        ----------
        *args : float or str
            Ordered arguments as listed in self._attributes.
            Takes precedence if same attribute is passed as arg and kwarg.
        **kwargs : float or str
            Named attributes.

        Returns
        -------
        None.

        """
        _default_attributes = self._default_attributes.copy()
        for attribute in _default_attributes:  # retain set attributes
            try:
                value = getattr(self, f"_{attribute}")
            except AttributeError:  # no underscore (read_txt)
                try:
                    value = getattr(self, attribute)
                except AttributeError:
                    value = None
            if value is not None:  # update attributes with namespace value
                _default_attributes[attribute] = value
        kwargs = {**_default_attributes, **kwargs}
        # if 'experiment' in kwargs:
        #     self._set_experiment(kwargs.pop('experiment'))
        for attribute in kwargs:
            value = kwargs[attribute]
            if value is not None:
                setattr(self, attribute, kwargs[attribute])
        # then set *args
        for attribute, value in zip(self._attributes, args):
            # if attribute == 'experiment':
            #     self._set_experiment(value)
            # else:
            setattr(self, attribute, value)

    @property
    def isset(self):
        """
        Return declaration status of all required inputs.

        Returns
        -------
        isset : pandas.Series(dtype=bool)
            Declration status of required inputs.

        """
        return pandas.Series(
            [getattr(self, f"_{attr}") is not None for attr in self._input_attributes],
            index=self._input_attributes,
            dtype=bool,
        )

    @property
    def isvalid(self):
        """
        Return overall status of required input attributes.

        Returns
        -------
        valid : bool
            Validation status, True if all _validation_attributes are set.

        """
        return np.array(self.isset).all()


if __name__ == "__main__":
    da = Attributes()
    da.attributes = "experiment"

    da2 = Attributes()
    print(da2.attributes, da.attributes)
