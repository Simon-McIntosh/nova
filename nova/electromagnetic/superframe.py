"""Configure superframe. Inherit DataArray for fast access else DataFrame."""

from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.dataarray import DataArray

SuperFrame = type('SuperFrame', (DataFrame,), {})
