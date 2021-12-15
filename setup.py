from setuptools import setup, find_packages
import versioneer

long_description = """..."""

extras_require = dict(test=['pytest'], develop=['pytest', 'spyder'])

setup_kwargs = dict(
    name                = 'nova',
    version             = versioneer.get_version(),
    description         = 'Equilibrium tools',
    license             = 'MIT',
    keywords            = 'equilibrium reconstruction plasma Biot Savart mesh free',
    author              = 'Simon McIntosh',
    author_email        = 'simon.mcintosh@iter.org',
    url                 = 'https://git.iter.org/projects/SCEN/repos/nova/',
    long_description    = long_description,
    packages            = find_packages(),
    classifiers         = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Fusion',
    ],
    cmdclass            = versioneer.get_cmdclass(),
    python_requires     = '>=3.9, <3.10',
    install_requires    = [
        'alphashape',
        #ansys-dpf-core==0.2.1
        #ansys-dpf-post
        #astropy
        'dask',
        'descartes',
        #gmsh
        #geojson,
        #line_profiler
        'meshio',
        'netCDF4',
        #networkx
        'nlopt',
        'numba',
        #numdifftools
        'numpy',
        #openpyxl
        'pandas',
        #plotly
        'pygeos',
        #pygmsh
        #pytest
        #python-dateutil
        'pyvista',
        'pyquaternion',
        'rdp',
        'seaborn',
        'scipy',
        'scikit-learn',
        'shapely',
        #spyder
        #tetgen
        #trimesh
        'vedo',
        #vtk
        'wheel',
        'xarray',
    ],
    extras_require     = extras_require,
    package_data       = {},
    include_package_data=False
)

setup(**setup_kwargs)
