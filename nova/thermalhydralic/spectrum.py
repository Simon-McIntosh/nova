from dataclasses import dataclass, field, InitVar
from typing import Union

import pandas


@dataclass
class Spectrum:
    """Container for sultan and twente spectrum data."""

    label: str
    data: InitVar[Union[pandas.DataFrame, dict]] = field(repr=False)
    columns: list = field(init=False, default_factory=list)
    # plot_kwargs: dict = field(default_factory=dict, repr=False)

    def __post_init__(self, data):
        self.columns = ["frequency", "B", "Q"]
        self.data = {}
        for column in data:
            print(data[column])


if __name__ == "__main__":
    spectrum = Spectrum("test", data)
