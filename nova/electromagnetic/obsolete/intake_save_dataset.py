# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:13:13 2021

@author: mcintos
"""

from warnings import warn

import intake
import yaml
import hvplot

from nova.electromagnetic.coilset import CoilSet
from nova.utilities.IO import pythonIO




class DataIO:
    """Abstract base class. Build, save and catalog DataArrays / DataSets."""

    def __init__(self, folder='ACLoss'):
        self.initialize_path(folder=folder)

    def initialize_path(self, folder='ACLoss'):
        """
        Generate file paths.

        Parameters
        ----------
        folder : str, optional
            Data folder. The default is 'ACLoss'.

        Returns
        -------
        None.

        """
        self.folder = folder
        self.directory = os.path.join(root_dir, f'data/{folder}')

    @staticmethod
    def remove(file):
        """Check for file, remove if found."""
        if os.path.isfile(file):
            os.remove(file)

    def _check_catalog_name(self, catalog_name=None):
        if catalog_name is None:
            catalog_name = self.folder
        if catalog_name[-5:] != '.yaml':
            catalog_name += '.yaml'
        return catalog_name

    def _check_catalog_file(self, catalog_name=None, create=False):
        """
        Check for intake catalog. Create if not found.

        Parameters
        ----------
        catalog_name : str, optional
            Catalog name set to self.folder if None. The default is None.

        Returns
        -------
        catalog_name : str
            Catalog name.

        """
        catalog_name = self._check_catalog_name(catalog_name)
        self.catalog_file = self._prepend(catalog_name)
        if not os.path.isfile(self.catalog_file):
            if create:
                catalog = intake.open_catalog()
                catalog.metadata = {'version': 1}
                catalog.save(self.catalog_file)  # create bare catalog
                catalog.close()
            else:
                raise FileNotFoundError(f'Catalog {self.catalog_file} '
                                        'not found.')
        return catalog_name

    def open_catalog(self, catalog_name=None):
        """
        Return intake catalog.

        Parameters
        ----------
        catalog_name : str, optional
            Catalog name, relative or full path. The default is None.

            - None : Catalog name defaults to self.folder

        Returns
        -------
        catalog : intake.catalog
            Intake catalog.

        """
        self._check_catalog_file(catalog_name)
        catalog = intake.open_catalog(self.catalog_file)
        return catalog

    def _prepend(self, file):
        """
        Return full file path.

        Parameters
        ----------
        file : str
            Filename, relative or full.

        Returns
        -------
        file : str
            Full file path.

        """
        if not os.path.split(file)[0]:  # prepend directory
            file = os.path.join(self.directory, file)
        return file

    def remove_catalog(self, catalog_name=None):
        """Remove catalog and associated data files."""
        catalog_name = self._check_catalog_name(catalog_name)
        catalog_file = self._prepend(catalog_name)
        if os.path.isfile(catalog_file):
            catalog = intake.open_catalog(catalog_file)
            for file in catalog:  # delete referanced files
                datafile = catalog[file].urlpath
                if os.path.isfile(datafile):
                    os.remove(datafile)
            os.remove(catalog_file)  # remove existing

    def close_catalog(self):
        """Close catalog."""
        self.catalog.close()

    def to_netcdf(self):
        """
        Save netCDF file, return file_hash.

        Returns
        -------
        file_hash : str
            Full file path.

        """
        # create temporary file
        file_tmp = os.path.join(self.directory, 'tmp.nc')
        self.remove(file_tmp)  # remove if already exists
        self.data.to_netcdf(file_tmp)  # save DataArray as netCDF file
        sha256_hash = pythonIO.hash_file(file_tmp, algorithm='sha256')  # hash
        file_hash = os.path.join(self.directory, sha256_hash)
        self.remove(file_hash)  # remove if already exists
        os.rename(file_tmp, file_hash)  # move file
        self.file_hash = file_hash

    def to_source(self, name, description, metadata):
        """
        Create intake source.

        Parameters
        ----------
        name : str
            Data source name.
        description : str
            Descriptive string.
        metadata : dict
            Source metadata.

        Returns
        -------
        None.

        """
        self.source = intake.open_netcdf(self.file_hash)
        self.source.name = name
        self.source.description = description
        self.source.metadata = metadata
        self.source.close()

    def to_catalog(self, catalog_name=None, replace=False):
        """
        Add source to catalog.

        Parameters
        ----------
        catalog_name : str, optional
            Catalog filename. The default is None.

        Returns
        -------
        None.

        """
        # load catalog.yaml file
        catalog_name = self._check_catalog_file(catalog_name, create=True)
        with open(self.catalog_file, 'r') as yamlfile:
            catalog = yaml.safe_load(yamlfile)
        # check for duplicates
        source_name = self.source.name
        if source_name in catalog['sources'].keys():
            datafile = catalog['sources'][source_name]['args']['urlpath']
            remove_file = os.path.isfile(datafile)  # if present
            if replace:  # and diffrent from linked referance
                remove_file &= (datafile != self.source.urlpath)
            if remove_file:
                os.remove(datafile)
            if not replace:
                warn(f'\n\nDuplicate source "{source_name}" '
                     f'found in catalog "{catalog_name}"\n'
                     'Save with diffrent name or to a diffrent catalog '
                     'or specify replace=True\n')
                return
        # load source.yaml
        source_file = self._prepend('tmp_source.yaml')
        with open(source_file, 'w') as f:
            f.write(self.source.yaml())
        with open(source_file, 'r') as yamlfile:
            source = yaml.safe_load(yamlfile)
        os.remove(source_file)  # remove temporary file
        # update catalog
        catalog['sources'].update(source['sources'])
        if catalog['sources']:
            with open(self.catalog_file, 'w') as yamlfile:
                yaml.safe_dump(catalog, yamlfile)

    def save(self, name, description='', replace=False, **metadata):
        """Save data to netCDF file."""
        self.to_netcdf()
        self.to_source(name, description, metadata)
        self.to_catalog(replace=replace)

    @property
    def catalog(self):
        """Return catalog, open if not present."""
        if not hasattr(self, '_catalog'):
            self._catalog = self.open_catalog()
        return self._catalog
