"""Manage polygon creation."""
from dataclasses import dataclass, field
from typing import Union

import shapely.geometry
import shapely.strtree
import numpy as np

from nova.frame.polygen import polygen, PolyFrame
from nova.frame.polygeom import PolyGeom
import matplotlib.pyplot as plt

# pylint:disable=unsubscriptable-object

    #def to_json(self, col='poly'):
    #    """Return col as json list with name, link and factor attrs."""
    #    return [dict(name=name, link=row['link'], factor=row['factor']) |
    #            json.loads(geojson.dumps(row[col]))
    #            for name, row in self.iterrows()]
