
Nova
----

Python toolkit for equilibrium generation / reconstruction and scenario design


Introduction
------------

**nova** is a Python package built to facilitate the generation and interpretation of magnetically confined plasma equilibria. This package extends [pandas](http://pandas.pydata.org) objects to add domain specific support for plasma filaments, electromagnetic coils, passive structures and circuits via `FrameSpace` types. Core electromagnetic calculations are implemented using a grid-free approach based on Biot-Savart integrals. Multi-dimensional data is represented internally using [xarray]( https://docs.xarray.dev). The results of expensive calculations are cached as [netCDF](https://www.unidata.ucar.edu/software/netcdf) files.


Dependancies 
------------
- [appdirs - Provides OS independent file system support]
- [click - Facilitates the implementation of CLI tools]
- [contourpy - Fast level-set contouring]
- [cython]
- [fsspec - Local / remote file system access]
- [gitpython - Git versioning]
- [netCDF4 - NetCDF data model]
- [networkx – For the construction of circuit DAGs]
- [numpy – Access to fast array-based operations] 
- [pandas – extended ‘DataFrame’ type to include domain specific functionality]
- [pytest – Test framework]
- [scipy – Scientific algorithms]
- [shapely – Manipulation of geometric objects] 
- [tqdm – Calculation progress monitor]
- [xarray – Representation of and operations on multi-dimensional data sets]
- [xxhash - Fast hashing to ensure calculations are only performed where necessary]


Install
-------

```sh
# development package using poetry with a pipx install
python3.10 -m pip install pipx-in-pipx
pipx install poetry 
pipx install tox 
poetry install -all-extras 

# run a single command using 
poetry run <command>

# or lanuch a poetry shell followed by the spyder IDE 
poetry shell 
spyder &
```

