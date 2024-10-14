
# nova: Python toolkit for equilibrium reconstruction and scenario design
[![image](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://git.iter.org/projects/EQ/repos/nova)
[![image](https://img.shields.io/badge/license-IO%20GIP-green)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)

## Introduction

**nova** is a Python package built to facilitate the generation and interpretation of magnetically confined plasma equilibria. This package extends [pandas](http://pandas.pydata.org) objects to add domain specific support for plasma filaments, electromagnetic coils, passive structures and circuits via `FrameSpace` types. Core electromagnetic calculations are implemented using a grid-free approach based on Biot-Savart integrals. Multi-dimensional data is represented internally using [xarray]( https://docs.xarray.dev). The results of expensive calculations are cached as [netCDF](https://www.unidata.ucar.edu/software/netcdf) files.


## Dependencies 

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


## Install

The NOVA package is PEP 517 compliant. Project dependancies are specified in the pyproject.toml file stored in the project's root. Base instalations may be perfomed using `pip` or `poetry`. Developlemnt installs should use `poetry`.

To install using pip (without an IDE)
```sh
# to install the NOVA base
pip install .

# optional extras may be specified using pip, such as the test enviroment 
pip install .[test]  

# to run the full test suite issue the following command from the project's root 
pytest
```

To install on SDCC (in advance of and EasyBuild module or for development work)
```sh 
# load IMAS module 
ml IMAS 
PYTHONPATH= PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --all-extras 

# run a single command using
poetry run <command> 

# launch Spyder 
poetry run spyder 

# launch Bokeh app
poetry run bokeh serve apps/pulsedesign
```

To install a development package using poetry (installed here prior to the nova package via pipx)
```sh
python3.10 -m pip install pipx-in-pipx
pipx install poetry 
pipx inject poetry poetry-plugin-export
pipx inject poetry "poetry-dynamic-versioning[plugin]"
poetry install --all-extras 
poetry run pre-commit install
 
# run a single command using 
poetry run <command>

# for example, to run the test suite use
poetry run pytest 

# or launch a poetry shell followed by the spyder IDE (with command returned back to the shell)
poetry shell 
spyder &
```

