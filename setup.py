from setuptools import setup, find_packages
import versioneer

long_description = """..."""

extras_require = dict(develop=['spyder', 'spyder-unittest', 'line_profiler', 'numdifftools', 'pytest-xdist'],
                      mesh=['gmsh', 'pygmsh', 'tetgen', 'trimesh'],
                      ansys=['ansys-dpf-core', 'ansys-dpf-post', 'openpyxl'],
                      thermofluids=['ftputil', 'coolprop'],
                      cuda=['cupy-cuda115'],
                      plan=['python-dateutil']
                      )

extras_require['full'] = [module for mode in extras_require for module in extras_require[mode] 
                          if mode not in ['thermofluids', 'cuda', 'plan']]

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
        'cython',
        'dask',
        'descartes',
        'fabric',
        'meshio',
        'netCDF4',
        'nlopt',
        'numba',
        'numpy',
        'pandas',
        'pylops',
        'pytest',
        'pyvista',
        'pyquaternion',
        'rdp',
        'seaborn',
        'scipy',
        'scikit-learn',
        'shapely',
        'vedo',
        'xarray',
        'xxhash',
    ],
    extras_require     = extras_require,
    package_data       = {},
    include_package_data=False
)

setup(**setup_kwargs)
