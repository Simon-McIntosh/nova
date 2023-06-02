"""Methods for managing Twente data."""
import os
import sys
from dataclasses import dataclass, InitVar

import numpy as np
import scipy
import scipy.signal
import pandas
import nlopt

from nova.definitions import root_dir
from nova.thermalhydralic.sultan.sultanio import SultanIO
import matplotlib.pyplot as plt
from nova.utilities.png_tools import data_mine


class TwenteFile:
    """Mixin for managing Twente data file structure."""

    @property
    def source_directory(self):
        """Return source directory."""
        return os.path.join(root_dir, "data/Twente")

    @property
    def image_directory(self):
        """Return full path of image directory."""
        return os.path.join(self.source_directory, self.experiment)

    @property
    def figurename(self):
        """Return figure name."""
        return f"{self.experiment}_{self.phase}"

    @property
    def filename(self):
        """Return filename."""
        return f"{self.figurename}_{self.index}"


@dataclass
class TwenteSource(TwenteFile, SultanIO):
    """Methods for digitizing Twente data."""

    experiment: str
    phase: str = "virgin"
    index: int = 0
    binary: InitVar[bool] = True

    def __post_init__(self, binary):
        """Load data."""
        if binary:
            self.data = self.load_data()
        else:
            self.data = self.read_data()

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return os.path.join(self.source_directory, "twentesource.h5")

    def _read_data(self):
        return self._mine_data()

    def _mine_data(self):
        """Extract data from image."""
        metadata = {
            "experiment": self.experiment,
            "phase": self.phase,
            "xlim": [0, 160],
            "ylim": [4, 20],
            "xlabel": "frequency",
            "xscale": 1e-3,
            "xunit": "Hz",
            "ylabel": "Q",
            "yscale": 1e-3 * 121.6 / 0.4,
            "yunit": "J/m",
            "B": 0.15,
        }
        # set experiment specific metadata
        if self.figurename == "CS_KAT_virgin":
            index_metadata = {
                "label": ["zero", "offset"][self.index],
                "Be": [0, 0.35][self.index],
            }
        elif self.figurename in ["CS_KAT_free", "CS_KAT_force"]:
            index_metadata = {
                "label": ["inital", "final"][self.index],
                "Be": 0,
                "cycles": [0, 30e3][self.index],
            }

        metadata |= index_metadata
        metalabel = ", ".join(
            [f"{attr}: {index_metadata[attr]}" for attr in index_metadata]
        )

        image_data = data_mine(
            self.image_directory,
            f"{self.experiment}_{self.phase}",
            metadata["xlim"],
            metadata["ylim"],
            title="\n".join([self.filename, metalabel]),
        )
        data = {}
        for axis in ["x", "y"]:
            data[metadata[f"{axis}label"]] = (
                metadata[f"{axis}scale"] * image_data.points[0][axis]
            )
        data = pandas.DataFrame(data)
        data.attrs = metadata
        return data

    def plot(self):
        """Plot data."""
        axes = plt.subplots(1, 1)[1]
        axes.plot(self.data["frequency"], self.data["Q"], "-o")


@dataclass
class TwentePost(TwenteFile, SultanIO):
    """Manage Twente data postprocess chain."""

    source: TwenteSource
    nsample: int = 100
    binary: InitVar[bool] = True
    verbose: bool = True

    def __post_init__(self, binary):
        """Load postprocess data."""
        if binary:
            self.read_data()
        else:
            self.read_data()
        self.Qhys = 0

    @property
    def frequency(self):
        """Return frequency, Hz."""
        return self.data.frequency

    @property
    def omega(self):
        """Return frequency, rad/s."""
        return self.data.omega

    @property
    def B(self):
        """Return field amplitude."""
        return self.data.attrs["B"]

    def _read_data(self):
        self._initialize_dataframe()
        self._interpolate()
        self.fit_polynomial()
        return self.data

    def _initialize_dataframe(self):
        """Initialize pandas DataFrame."""
        self.data = pandas.DataFrame(
            index=range(self.nsample), columns=["frequency", "omega", "Q", "Qdot"]
        )
        self.data.attrs = self.source.data.attrs
        self.data.attrs["nsample"] = self.nsample

    def _interpolate(self):
        """Interpolate source data."""
        self.data.loc[:, "frequency"] = np.logspace(
            np.log10(self.source.data.frequency.iloc[0]),
            np.log10(self.source.data.frequency.iloc[-1]),
            self.nsample,
        )
        self.data.loc[:, "omega"] = 2 * np.pi * self.data.frequency
        self.data.loc[:, "Q"] = scipy.interpolate.interp1d(
            self.source.data.frequency,
            self.source.data.Q,
            bounds_error=False,
            fill_value="extrapolate",
        )(self.data.frequency)

    def frequency_matrix(self, order, index):
        """
        Return frequency matrix for lest squares polynomial fit.

        Parameters
        ----------
        order : int
            Polynomial order.
        index : slice
            Index.

        Returns
        -------
        frequency : array-like
            Frequency array.
        frequency_matrix : array-like, (n, order)
            Frequency matrix

        """
        # frequency = self.data.loc[index, 'frequency'].to_numpy().reshape(-1, 1)
        frequency = self.source.data.loc[index, "frequency"].to_numpy().reshape(-1, 1)
        matrix = np.concatenate([frequency**i for i in range(order + 1)], axis=1)
        return frequency, matrix

    def fit_polynomial(self, order=2, index=slice(None), plot=False):
        """Fit polynomial to heat data."""
        # extract data
        frequency, matrix = self.frequency_matrix(order, index)
        # heat = self.data.loc[index, 'Q']
        heat = self.source.data.loc[index, "Q"]
        # fit coefficients
        coefficients = np.linalg.lstsq(matrix, heat, rcond=None)[0]
        # re-construct
        if index != slice(None):
            matrix = self.frequency_matrix(order, slice(None))[1]
        # self.data.loc[:, 'Qpoly'] = matrix @ coefficients
        self.data.attrs["polynomial"] = {
            "order": order,
            "index": index,
            "coefficients": coefficients,
        }
        if plot:
            self.plot_polynomial()
        return coefficients

    def plot_polynomial(self):
        """Plot polynomial fit."""
        axes = plt.subplots(1, 1)[1]
        polynomial = self.data.attrs["polynomial"]
        frequency, matrix = self.frequency_matrix(polynomial["order"], slice(None))
        polynomial_fit = matrix @ polynomial["coefficients"]
        axes.plot(self.frequency, self.data.Q, "o")
        axes.plot(frequency, polynomial_fit, "--", color="gray")
        axes.plot(
            frequency[polynomial["index"]],
            polynomial_fit[polynomial["index"]],
            "--",
            color="C3",
        )

    def fit_transfer_function(self, zeros, poles, gain, Qhys):
        """Fit transfer function to heat data."""
        nzeros = len(zeros)

        def model(vector):
            """Return lti model."""
            zeros = 10 ** np.array(vector[:nzeros])
            poles = 10 ** np.array(vector[nzeros:-2])
            """
            order = 20
            step = 0.1
            cutoff = vector[0]
            poles = np.logspace(cutoff,
                                cutoff + step*(order-1),
                                order)
            zeros = np.logspace(cutoff + step/2,
                                cutoff + step*(order-1 + 0.5),
                                order)
            #zeros = poles[0] + np.diff(poles)
            #poles = np.array([cutoff * k**(2*i) for i in range(order)])
            #zeros = np.array([cutoff * k**(2*i+1) for i in range(order)])
            """
            poles = np.array([poles[0] + 1j * poles[1], poles[0] - 1j * poles[1]])
            dcgain = vector[-2]
            gain = np.prod(poles) / np.prod(zeros) * dcgain
            return scipy.signal.ZerosPolesGain(-zeros, -poles, gain)

        def model_heat(vector):
            """
            Return heat magnitude response.

            Parameters
            ----------
            vector : array-like
                Optimization vector.

            Returns
            -------
            Qdot : array-like
                Heat response.

            """
            system = model(vector)
            bode = scipy.signal.bode(system, w=self.data.omega)[1]
            return 10 ** (bode / 20)

        def model_error(vector):
            """
            Return L2norm model error.

            Parameters
            ----------
            vector : array-like
                Optimization vector.

            Returns
            -------
            error : float
                L2-norm error.

            """
            Prms = model_heat(vector)
            # self.Qhys = vector[-1]  # update Qhys estimate
            L2norm = np.linalg.norm(self.Prms - Prms, axis=0)
            return 1e3 * L2norm / self.nsample

        def model_update(vector, grad):
            """Return L2norm error and evaluate gradient in-place."""
            err = model_error(vector)
            if self.verbose:
                sys.stdout.write(f"\r{err}")
                sys.stdout.flush()
            if len(grad) > 0:
                grad[:] = scipy.optimize.approx_fprime(vector, model_error, 1e-6)
            return err

        def model_bound(vector):
            """Return Qdot inequality constraint."""
            Prms = model_heat(vector)
            return np.max(self.Prms - Prms)

        def bound_update(result, vector, grad):
            """Set bound constraint."""
            result[:] = model_bound(vector)
            if len(grad) > 0:
                grad[:] = scipy.optimize.approx_fprime(vector, model_bound, 1e-6)

        def get_opt(algorithum, zeros, poles):
            parameter_number = len(zeros) + len(poles) + 2
            lower_bounds = [-3 for __ in zeros + poles] + [1e-12, 0]
            upper_bounds = [0.5 for __ in zeros + poles] + [1e12, 1e12]
            opt = nlopt.opt(f"LN_{algorithum}", parameter_number)
            opt.set_initial_step(0.1)
            opt.set_min_objective(model_update)
            opt.set_lower_bounds(lower_bounds)
            opt.set_upper_bounds(upper_bounds)
            opt.set_ftol_rel(1e-8)
            return opt

        inital_vector = zeros + poles + [gain, Qhys]

        opt = get_opt("BOBYQA", zeros, poles)
        vector = opt.optimize(inital_vector)

        # opt = get_opt('COBYLA', zeros, poles)
        # opt.add_inequality_mconstraint(bound_update, [0])
        # opt.set_ftol_rel(1e-3)
        # vector = opt.optimize(inital_vector)

        return vector, model(vector)

    @property
    def Qdot(self):
        """Return heat rate."""
        return self.data.Qdot

    @property
    def Qhys(self):
        """Manage hysterisis heat."""
        return self._Qhys

    @Qhys.setter
    def Qhys(self, Qhys):
        self._Qhys = Qhys
        Qdot = self.data.Q - Qhys
        Qdot /= 2 * np.pi * self.data.omega * self.B**2
        self.data.loc[:, "Qdot"] = Qdot

    @property
    def Prms(self):
        """Return rms gain."""
        return self.data.Qdot**0.5

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return os.path.join(self.source_directory, "twentepost.h5")

    @property
    def experiment(self):
        """Return experiment name."""
        return self.source.experiment

    @property
    def phase(self):
        """Return experiment phase."""
        return self.source.phase

    @property
    def index(self):
        """Return esperiment index."""
        return self.source.index

    def plot(self):
        poly = self.fit_polynomial(2, index=slice(3), plot=False)
        self.Qhys = 0  # 1.05*poly[0]

        axes = plt.subplots(1, 1)[1]
        vector, system = self.fit_transfer_function(
            [], [-1.5, -0.4], self.Prms[0], self.Qhys
        )
        print("\n", vector)
        frequency = np.logspace(-3, 0.5)
        #
        # frequency = self.frequency
        bode = scipy.signal.bode(system, w=2 * np.pi * frequency)[1]
        Prms_model = 10 ** (bode / 20)
        axes.plot(frequency, Prms_model, "-")

        axes.plot(self.data.frequency, self.Prms, ".-")
        plt.despine()
        axes.set_xscale("log")
        axes.set_yscale("log")
        print(self.Qhys)

        return system


if __name__ == "__main__":
    source = TwenteSource("CS_KAT", phase="virgin", index=0, binary=True)
    post = TwentePost(source, binary=False)

    system = post.plot()

    omega = post.omega[0]
    cycles = 50
    t = np.linspace(0, cycles * 2 * np.pi / omega, 30 * cycles)
    y = np.sin(omega * t)
    p = scipy.signal.lsim(system, y, t)[1]

    # plt.figure()
    # plt.plot(t, y)
    # plt.plot(t, p)
    # print(2*np.mean(p**2))

    """
    for experiment in ['CSJA_7', 'CSJA_8', 'CSJA12', 'CSJA13']:
        fluid = FluidResponse(experiment, 0, 'Left')
        omega, gain = fluid.response(2)
        plt.plot(omega/(2*np.pi), gain, 'o-', label=experiment)
    """

    '''

    def process(self):
        rawdata = {}
        frequency = []
        for series, data_points in zip(metadata['columns'], image_data.points):
            rawdata[series] = {}
            for axis in ['x', 'y']:
                rawdata[series][metadata[f'{axis}label']] = \
                    metadata[f'{axis}scale']*data_points[axis]
            rawdata[series]['Qdot'] -= rawdata[series]['Qdot'][0]
            rawdata[series]['Qdot'] /= rawdata[series]['frequency']
            rawdata[series]['Qdot'] /= (2*np.pi)**2
            rawdata[series]['Qdot'] /= metadata['B']**2
            frequency.extend(rawdata[series]['frequency'])
        # interpolate data
        data = {'frequency': np.logspace(
            np.log10(np.min(frequency)), np.log10(np.max(frequency)),
            metadata['nsample'])}
        for series in rawdata:
            data[series] = scipy.interpolate.interp1d(
                np.log10(rawdata[series]['frequency']),
                rawdata[series]['Qdot'],
                bounds_error=False, fill_value='extrapolate')(
                    np.log10(data['frequency']))

    def plot(self):
        """Plot data."""
        axes = plt.subplots(1, 1)[1]

        frequency = np.linspace(self.data['frequency'].iloc[0],
                                self.data['frequency'].iloc[-1], 50)
        Qdot = scipy.interpolate.interp1d(
            self.data['frequency'], self.data['Qdot'])(frequency)
        frequency_matrix = np.concatenate(
            [np.ones((len(frequency), 1)),
             frequency.reshape(-1, 1),
             frequency.reshape(-1, 1)**2], axis=1)

        poly = np.linalg.lstsq(frequency_matrix, Qdot, rcond=None)[0]

        Qfit = frequency_matrix @ poly
        Qfit -= poly[0]
        Qfit /= frequency
        Qfit /= 4*np.pi**2
        Qfit /= self.data.attrs['B']**2

        self.data['Qdot'] -= poly[0]
        self.data['Qdot'] /= self.data['frequency']
        self.data['Qdot'] /= 4*np.pi**2
        self.data['Qdot'] /= self.data.attrs['B']**2

        #frequency_matrix = np.concatenate(
        #    [(2*np.pi*frequency).reshape(-1, 1)**2,
        #     np.ones((len(frequency), 1))], axis=1)
        #poly_tf = np.linalg.lstsq(frequency_matrix, Qfit**-2, rcond=None)[0]

        frequency_matrix = np.concatenate(
            [(2*np.pi*self.data['frequency'].values).reshape(-1, 1)**2,
             np.ones((len(self.data['frequency']), 1))], axis=1)
        poly_tf = np.linalg.lstsq(frequency_matrix, self.data['Qdot']**-2, rcond=None)[0]

        poly_tf = np.sqrt(poly_tf)
        gain = 1/poly_tf[0]
        cutoff = poly_tf[1]/poly_tf[0]
        dcgain = gain/cutoff
        print(gain, cutoff, dcgain)

        system = scipy.signal.ZerosPolesGain([], [-cutoff], gain)
        bode = scipy.signal.bode(system, w=2*np.pi*frequency)[1]



        print((poly[1]-poly[0]) / (4*np.pi**2*self.data.attrs['B']**2))
        print(scipy.signal.step(system)[1][-1])
        print(cutoff/(2*np.pi))

        axes.plot(self.data['frequency'], self.data['Qdot'], '-o')  # 0.39
        axes.plot(frequency, Qfit, '--')
        axes.plot(frequency, 10**(bode/20), '-')
        plt.despine()
        axes.set_xscale('log')
        axes.set_yscale('log')
    '''
