from setuptools import setup, find_packages
import versioneer

long_description = """ """ 

extras_require = dict(
                      ansys=['ansys-dpf-core', 'ansys-dpf-post', 'openpyxl'],
                      cuda=['cupy-cuda115'],
                      develop=['spyder', 'spyder-unittest', 'line_profiler', 'numdifftools', 'pytest-xdist'],
                      mesh=['gmsh', 'pygmsh', 'tetgen', 'trimesh'],
                      optimize=['nlopt', 'pygmo'],
                      plan=['python-dateutil'],
                      test=['pytest', 'lizard', 'asv', 'virtualenv'],
                      thermofluids=['ftputil', 'coolprop', 'tables', 'xlrd']
                      )

extras_require['full'] = [module for mode in extras_require for module in extras_require[mode] 
                          if mode not in ['cuda', 'plan', 'optimize', 'mesh']]

setup_kwargs = dict(
    name                = 'nova',
    version             = versioneer.get_version(),
    description         = 'Equilibrium tools',
    license             = 'ITER GIP',
    keywords            = 'equilibrium reconstruction plasma Biot Savart mesh free',
    author              = 'Simon McIntosh',
    author_email        = 'simon.mcintosh@iter.org',
    url                 = 'https://git.iter.org/projects/EQ/repos/nova/',
    long_description    = long_description,
    packages            = find_packages(),
    include_package_data= True,
    package_data        = {},
    classifiers         = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Fusion',
    ],
    cmdclass            = versioneer.get_cmdclass(),
    python_requires     = '>=3.10',
    install_requires    = [
        'alphashape',
        'appdirs',
        'click',
        'cython',
        'dask',
        'descartes',
        'fabric',
        'fortranformat',
        'gitpython',
        'meshio',
        'netCDF4',
        'numba',
        'numpy',
        'pandas',
        'pylops',
        'pyperf',
        'pyvista',
        'pyquaternion',
        'rdp',
        'seaborn',
        'scipy',
        'scikit-learn',
        'shapely',
        'vedo',
        'wheel',
        'xarray',
        'xxhash',
    ],
    extras_require     = extras_require,
    entry_points={'console_scripts': [
                      'benchmark = nova.scripts.benchmark:benchmark']},
)

setup(**setup_kwargs)
