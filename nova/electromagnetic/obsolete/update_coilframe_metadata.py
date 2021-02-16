# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:39:09 2021

@author: mcintos
"""

    _metadata = ['_required_columns',
                 '_additional_columns',
                 '_default_attributes',
                 '_coildata_attributes',
                 '_dataframe_attributes']

    def _update_coilframe_metadata(self, **coilframe_metadata):
        'extract and update coilframe_metadata'
        mode = coilframe_metadata.pop('mode', 'append')  # [overwrite, append]
        for key in coilframe_metadata:
            if mode == 'overwrite':
                null = [] if key[1:] in ['required_columns',
                                         'additional_columns',
                                         'dataframe_attributes'] else {}
                setattr(self, key, null)
            value = coilframe_metadata.get(key, None)
            if value is not None:
                if key == '_required_columns':
                    self._required_columns = value  # overwrite
                elif key == '_additional_columns':
                    for v in value:
                        if v not in getattr(self, key):
                            getattr(self, key).append(v)
                elif key == '_default_attributes':
                    for k in value:  # set/overwrite dict
                        self._default_attributes[k] = value[k]
                elif key in self._default_attributes:
                    self._default_attributes[key] = value
                elif key == '_dataframe_attributes':
                    self.dataframe_attributes = value
                elif key == '_coildata_attributes':
                    self.coildata_attributes = value
                elif key in self._coildata_attributes:
                    self.coildata_attributes = {key: value}

    @property
    def coilframe_metadata(self):
        'extract coilframe_metadata attributes'
        self._coildata_attributes = self.coildata_attributes
        return {key: getattr(self, key) for key in self._metadata}

    @coilframe_metadata.setter
    def coilframe_metadata(self, coilframe_metadata):
        'update coilframe_metadata attributes'
        self._update_coilframe_metadata(**coilframe_metadata)
        self.coildata_attributes = self._coildata_attributes