"""Build regressors based on two coil TF central composite simulations."""
from dataclasses import dataclass, field
import os

import numpy as np
import numpy.typing as npt
import pickle
import pyvista as pv
import xarray

from nova.structural.ansyspost import AnsysPost
from nova.structural.datadir import AnsysDataDir
from nova.utilities.time import clock

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split


@dataclass
class TFC2_DoF(AnsysDataDir):
    """Load central composite design."""

    folder: str = 'TFC2_DoE'
    subset: str = 'TF1_CASE'
    reference_current: dict[str, float] = field(
        default_factory=lambda: dict(CS3U=4.829, CS2U=4.048, CS1=20.804,
                                     CS2L=10.297, CS3L=-5.301,
                                     PF1=-5.732, PF2=2.883, PF3=5.808,
                                     PF4=4.776, PF5=7.813, PF6=-16.941,
                                     Plasma=-14.999))
    datasets: list[str] = field(
        default_factory=lambda: ['m3', 'm2', 'm1', 'p1', 'p2', 'p3'])
    attributes: list[str] = field(default_factory=lambda: ['disp', 'vm'])
    reference_loadcase: int = 5
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load dataset."""
        super().__post_init__()
        self.n_coils = len(self.reference_current)
        self.n_datasets = len(self.datasets)
        self.load()

    @property
    def vtk_file(self):
        """Return vtk filename."""
        return os.path.join(self.directory, f'{self.subset}.vtk')

    def load(self):
        """Load vtk mesh file."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.load_ansys()
        self.load_xarray()

    def load_ansys(self):
        """Load ansys data."""
        self.initalize_mesh()
        tick = clock(len(self.datasets), header='Reading ansys datasets')
        for dataset in self.datasets:
            self.load_dataset(dataset)
            tick.tock()
        self.mesh.save(self.vtk_file)

    def load_dataset(self, dataset):
        """Load single data and store tared values as node arrays."""
        mesh = AnsysPost(self.folder, dataset, self.subset).mesh
        for attr in self.attributes:
            for i in range(self.n_coils):
                self.mesh[f'{dataset}_{attr}-{i}'] = \
                    mesh[f'{attr}-{i+self.reference_loadcase+1}'] - \
                    mesh[f'{attr}-{self.reference_loadcase}']
                if attr == 'vm':
                    self.mesh[f'{dataset}_{attr}-{i}'] *= 1e-6

    def initalize_mesh(self):
        """Generate pyvista mesh."""
        self.mesh = AnsysPost(self.folder, self.datasets[0], self.subset).mesh
        self.mesh.clear_arrays()

    def plot_slice(self, loadcases=[6, 7, 8, 9, 10, 11], factor=750):
        """Plot dataset slice."""
        plotter = pv.Plotter(shape=(len(loadcases)+1, self.n_datasets),
                             border=False, window_size=[1400, 1400])
        sargs = dict(title_font_size=18, label_font_size=18,
                     shadow=False, n_labels=2, vertical=False, width=0.5)

        for i, dataset in enumerate(self.datasets):
            for j, loadcase in enumerate(loadcases):
                plotter.subplot(j, i)
                warp = self.mesh.warp_by_vector(f'{dataset}_disp-{loadcase}',
                                                factor=factor)
                plotter.add_mesh(warp, scalars=f'{dataset}_vm-{loadcase}',
                                 scalar_bar_args=sargs)
                coil = list(self.reference_current)[loadcase]
                sign = '+' if dataset[0] == 'p' else '-'
                title = f'{coil} {sign}{dataset[1]}MA'
                plotter.add_title(title, font_size=12)
        plotter.link_views()
        plotter.camera_position = 'xz'
        plotter.camera.zoom(1.5)
        plotter.show()

    @property
    def n_trials(self):
        """Return trial number."""
        return self.n_coils*self.n_datasets

    def get_current(self):
        """Return current array."""
        current = np.zeros((self.n_trials, self.n_coils))
        current[:] = list(self.reference_current.values())
        for i, dataset in enumerate(self.datasets):
            for j, coil in enumerate(self.reference_current):
                sign = 1 if dataset[0] == 'p' else -1
                current[i*self.n_coils + j, j] += sign*int(dataset[1])
        return current

    def get_data(self, attr):
        """Extract data from vtk structure."""
        if attr == 'vm':
            data = np.zeros((self.n_trials, self.mesh.n_points))
        else:
            data = np.zeros((self.n_trials, self.mesh.n_points, 3))
        for i, dataset in enumerate(self.datasets):
            for j, coil in enumerate(self.reference_current):
                data[i*self.n_coils + j] = self.mesh[f'{dataset}_{attr}-{j}']
        return data

    def load_xarray(self):
        """Save vtk data as xarray dataset."""
        self.data = xarray.Dataset(
            coords=dict(coil=list(self.reference_current),
                        trial=range(self.n_coils*self.n_datasets),
                        points=range(self.mesh.n_points),
                        vector=['x', 'y', 'z']))
        self.data['current'] = (['trial', 'coil'], self.get_current())
        self.data['vm'] = (['trial', 'points'], self.get_data('vm'))
        self.data['disp'] = (['trial', 'points', 'vector'],
                             self.get_data('disp'))


@dataclass
class Regressor:
    """Manage scikit-learn regressor."""

    data: xarray.Dataset = field(repr=False)
    model: str
    target: str = 'vm'
    feature: str = 'current'
    target_index: npt.ArrayLike = field(init=False, repr=False)
    train: npt.ArrayLike = field(init=False, repr=False)
    test: npt.ArrayLike = field(init=False, repr=False)
    regressor: BaseEstimator = field(init=False, repr=False)

    def __post_init__(self):
        """Load regressor."""
        self.load_data()
        self.train_model()

    '''
    @property
    def pk_file(self):
        """Return pickle filename."""
        return os.path.join(self.directory, f'{self.subset}.pk')

    def load(self):
        """Load sklear regressor."""
        try:
            with open(self.pk_file, 'rb') as pk_file:
                self.regressor = pickle.load(pk_file)
        except FileNotFoundError:
            pass
            self.load_ansys()

    def store(self):
        """Store sklearn regressor."""
        with open(self.pk_file, 'wb') as pk_file:
            pickle.dumps(self.regressor, pk_file)
    '''

    def train_model(self):
        """Fit regressor."""
        self.regressor = self.initialize_model()
        self.regressor.fit(*self.train)

    def initialize_model(self):
        """Return initialized regresion model."""
        if self.model == 'linear':
            return LinearRegression()
        return KernelRidge(alpha=0, kernel='rbf')

    def load_data(self):
        """Load train and test datasets."""
        self.target_index = ~np.isnan(self.data[self.target][0].values)
        data = train_test_split(
            self.data[self.feature].values,
            self.data[self.target].values[:, self.target_index],
            random_state=1, test_size=0.25)
        self.train = data[::2]
        self.test = data[1::2]

    def validate(self, cv=2):
        """Print validation metrics."""
        score = cross_val_score(self.regressor, *self.train, cv=cv)
        print(np.mean(score), np.std(score))
        return score

    def score(self):
        """Print test data score."""
        score = self.regressor.score(*self.test)
        return score


    def predict(self, feature_vector):
        """Return regressor prediction."""
        values = np.full(self.target_index.shape, 0, dtype=float)
        values[self.target_index] = self.regressor.predict(
            feature_vector.reshape(1, -1))[0]
        return values


@dataclass
class Simulator:

    subset: str
    design: TFC2_DoF = field(init=False, repr=False)
    models: dict = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        self.design = TFC2_DoF(subset=self.subset)
        self.mesh = self.design.mesh.copy()
        self.mesh.clear_arrays()
        self.load_models()

    def load_models(self):
        self.models = {}
        for model in ['linear', 'kridge']:
            for attr in ['disp', 'vm']:
                self.models[f'{model}_{attr}'] = \
                    Regressor(self.design.data.copy(deep=True), model, attr)

    def predict(self, feature_vector):
        for attr in self.models:
            self.mesh[attr] = self.models[attr].predict(feature_vector)

    def plot(self, model, plotter=None, factor=1e3, show=True, clim=None):
        if plotter is None:
            plotter = pv.Plotter()
        warp = self.mesh.warp_by_scalar(f'{model}_disp', factor=factor)
        plotter.add_mesh(warp, scalars=f'{model}_vm', clim=clim)
        if show:
            plotter.show()

    def plot_pair(self, clim=None):
        plotter = pv.Plotter(shape=(2, 2), border=False)
        sargs = dict(title_font_size=18, label_font_size=18,
                     shadow=False, n_labels=3, vertical=False, width=0.5)
        plotter.subplot(0, 0)
        self.plot('linear', plotter, show=False)
        score = self.models['linear_vm'].score()
        plotter.add_title(f'{1e2*score:1.1f}%', font_size=12)
        plotter.subplot(0, 1)
        self.plot('kridge', plotter, show=False)
        score = self.models['kridge_vm'].score()
        plotter.add_title(f'{1e2*score:1.1f}%', font_size=12)
        plotter.link_views()
        plotter.camera_position = 'xz'
        plotter.camera.zoom(2.0)
        plotter.show()

    def plot_compare(self, trial, clim=None):
        self.mesh['ground_truth'] = self.design.data['vm'][trial]
        current = self.design.data['current'][trial].values
        for model in ['linear', 'kridge']:
            self.mesh[f'predict_{model}'] = \
                self.models[f'{model}_vm'].predict(current)

        plotter = pv.Plotter(shape=(2, 3))
        #plotter.subplot(0, 0)
        plotter.add_mesh(self.mesh, scalars='ground_truth', clim=clim)
        plotter.add_title('ground_truth', font_size=12)

        plotter.subplot(0, 1)
        plotter.add_mesh(self.mesh, scalars=f'predict_linear', clim=clim)
        plotter.add_title('linear', font_size=12)

        plotter.subplot(0, 2)
        plotter.add_mesh(self.mesh, scalars=f'predict_kridge', clim=clim)
        plotter.add_title('kridge', font_size=12)

        plotter.link_views()
        plotter.camera_position = 'xz'
        plotter.camera.zoom(1.0)
        plotter.show()



if __name__ == '__main__':

    #subset = 'TF1_Case'
    subset = 'TF1_GS_TF_LEG'
    #subset = 'TF1_BOTTOM_SHEAR_KEYS'
    #subset = 'TF1_WP'
    #subset = 'TF1_IOIS_DOWN_PINS'

    current = np.array(list(TFC2_DoF().reference_current.values()))
    #current[8] += 5

    sim = Simulator(subset)
    sim.predict(current)
    #sim.design.plot_slice(loadcases=[0, 8, 11])

    sim.plot_pair()

    #sim.plot_compare(50, clim=None)
    '''
    #, datasets=['m1', 'p1'])
    #dof.plot_slice(loadcases=[9], factor=50)

    reg = Regressor(dof.data, False, 'disp')

    reg.score()



    warp = dof.mesh.warp_by_vector('disp', factor=factor)
    dof.mesh['predict'] = reg.predict(current)
    dof.mesh.plot(scalars='predict')
    '''
