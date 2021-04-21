# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:39:58 2021

@author: mcintos
"""

    '''
    @property
    def boundary(self):
        """
        Manage plasma limit boundary.

        Parameters
        ----------
        boundary : array-like, shape(4,) or array-like, shape(n, 2) or Polygon
            External plasma boundary (limit).
            Coerced as a positively oriented curve.

            - array-like, shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        Raises
        ------
        IndexError
            Malformed bounding box, shape is not (4,).

            Malformed bounding loop, shape is not (n, 2).

        Returns
        -------
        plasma_boundary : Polygon
            Plasma limit boundary.

        """
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if not isinstance(boundary, shapely.geometry.Polygon):
            boundary = np.array(boundary)  # to numpy array
            if boundary.ndim == 1:   # limit bounding box
                if len(boundary) == 0:
                    return
                elif len(boundary) == 4:
                    polygon = shapely.geometry.box(*boundary[::2],
                                                   *boundary[1::2])
                else:
                    raise IndexError('malformed bounding box\n'
                                     f'boundary: {boundary}\n'
                                     'require [xmin, xmax, zmin, zmax]')
            elif boundary.ndim == 2 and (boundary.shape[0] == 2 or
                                         boundary.shape[1] == 2):  # loop
                if boundary.shape[1] != 2:
                    boundary = boundary.T
                polygon = shapely.geometry.Polygon(boundary)
            else:
                raise IndexError('malformed bounding loop\n'
                                 f'shape(boundary): {boundary.shape}\n'
                                 'require (n,2)')
        else:
            polygon = boundary
        # orient polygon
        polygon = shapely.geometry.polygon.orient(polygon)
        self._boundary = polygon
        #if 'plasmagrid' in self.biot_instances:
        #    self.plasmagrid.plasma_boundary = polygon
    '''