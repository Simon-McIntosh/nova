#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:43:35 2021

@author: mcintos
"""

    def validate(self):
        """Repair polygon if not valid."""
        if not polygon.is_valid:
            polygon = pygeos.creation.polygons(loop)
            polygon = pygeos.constructive.make_valid(polygon)
            area = [pygeos.area(pygeos.get_geometry(polygon, i))
                    for i in range(pygeos.get_num_geometries(polygon))]
            polygon = pygeos.get_geometry(polygon, np.argmax(area))
            polygon = shapely.geometry.Polygon(
                pygeos.get_coordinates(polygon))