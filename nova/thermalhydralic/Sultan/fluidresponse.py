
from dataclasses import dataclass
from typing import Union
import sys

import pandas
import pygmo
import numpy as np
import scipy
import nlopt

from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.thermalhydralic.sultan.fluidprofile import FluidProfile
from nova.thermalhydralic.sultan.sultanio import SultanIO
from nova.thermalhydralic.sultan.remotedata import FTPData
from nova.thermalhydralic.sultan.campaign import Campaign
from nova.utilities.pyplot import plt
from nova.utilities.time import clock


@dataclass
class FluidResponse(SultanIO):
    """Manage fluid response data."""

    experiment: str
    phase: Union[int, str]
    side: str
    order: Union[int, list[int]] = 6
    threshold: float = 0

    def __post_init__(self):
        """Create fluidprofile instance."""
        trial = Trial(self.experiment, self.phase)
        sample = Sample(trial, _side=self.side)
        self.fluidprofile = FluidProfile(sample, self.order, self.threshold,
                                         verbose=False)
        self.coefficents = self.load_data()
        self._load_plan()

    def _load_plan(self):
        self.plan = self.sample.trial.plan.droplevel(1, axis=1)
        self.plan.drop(columns='File', inplace=True)

    @property
    def profile(self):
        """Return profile instance."""
        return self.fluidprofile.profile

    @property
    def sample(self):
        """Return sample instance."""
        return self.profile.sample

    @property
    def database(self):
        """Return database instance."""
        return self.sample.trial.database

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return self.database.binary_filepath('fluidresponse.h5')

    @property
    def filepath(self):
        """Return full path of source datafile, read-only."""
        return self.database.datafile(self.filename)

    @property
    def filename(self):
        """Return datafile filename."""
        filename = f'{self.sample.trial.campaign.experiment}'
        filename += f'_{self.sample.trial.name}_{self.sample.side}'
        return filename

    @property
    def shot_coefficents(self):
        """Return profile, model and fit shot coefficents."""
        return pandas.concat([self.profile.coefficents,
                              self.fluidprofile.coefficents])

    def _read_data(self):
        """Refit fluid model to waveform data."""
        coefficents = pandas.DataFrame(index=range(self.sample.samplenumber),
                                       columns=self.shot_coefficents.index,
                                       dtype='float')
        tick = clock(self.sample.samplenumber,
                     header=f'Calculating {self.filename} fluid response.')
        for shot in self.sample.sequence():
            coefficents.iloc[shot] = self.shot_coefficents
            tick.tock()
        return coefficents.astype('float')

    def _sort_monotonic(self, index):
        points = [(-np.log10(frequency), -np.log10(value))
                  for frequency, value in
                  zip(self.coefficents['frequency'][index],
                      self.coefficents['dcgain'][index])]
        pareto_index = pygmo.non_dominated_front_2d(points)
        return self.coefficents['frequency'].index[index][pareto_index][::-1]

    def response(self, Be):
        index = (self.plan['Be'] == Be) & (self.plan['Isample'] == 0)
        pareto_index = self._sort_monotonic(index)
        frequency = 2*np.pi * self.coefficents['frequency'][pareto_index].to_numpy()
        gain = self.coefficents['steadystate'][pareto_index].to_numpy()
        return frequency, gain

    def plot(self, Be, dcgain_limit=1e6):
        """Plot frequency response."""


        '''
        plt.plot(2*self.coefficents['frequency'][pareto_index],
                   self.coefficents['steadystate'][pareto_index], '-',
                   label='model')
        plt.plot(2*self.coefficents['frequency'][index],
                   self.coefficents['maximum_value'][index], label='data')
        '''
        #plt.legend()

        _frequency, _gain = self.response(Be)
        frequency = np.logspace(np.log10(_frequency[0]),
                                np.log10(_frequency[-1]), 40)
        gain = scipy.interpolate.interp1d(
            np.log10(_frequency), _gain)(np.log10(frequency))

        #gain = gain**2

        #gain /= frequency
        #frequency *= 100
        #plt.plot(frequency, gain)

        '''
        order = 4
        frequency_matrix = np.concatenate(
            [(1j * frequency).reshape(-1, 1)**i
             for i in np.arange(order)[::-1]], axis=1)

        def error(x, grad):
            system_gain = x[0]
            model_gain = system_gain / np.absolute(frequency_matrix @ x[1:])
            error = np.linalg.norm(model_gain - gain)
            sys.stdout.write(f'\r{error}')
            sys.stdout.flush()
            return error
        '''

        #dBgain = 20*np.log10(gain)

        nzero = 0
        npole = 2
        dimension = nzero+npole+1

        def field_model(x):
            zeros = x[:nzero]
            poles = -np.abs(x[-npole-1:-1])
            dcgain = x[-1]
            system_gain = dcgain * np.prod(poles) / np.prod(zeros)
            #print(_x, zeros, poles, gain)
            return scipy.signal.ZerosPolesGain(zeros, poles, system_gain)

        def model_gain(x):
            dBgain = scipy.signal.bode(field_model(x), frequency)[1]
            return 10**(dBgain / 40)

        def model_error(x):
            error = np.linalg.norm(np.log10(model_gain(x)) -
                                   np.log10(gain))
            #error = np.linalg.norm(model_gain(x) - gain)
            return error

        def error(x, grad):
            error = model_error(x)
            sys.stdout.write(f'\r{error}')
            sys.stdout.flush()
            if len(grad) > 0:
                grad[:] = scipy.optimize.approx_fprime(x, model_error, 1e-6)
            return error

        #opt = nlopt.opt(nlopt.LN_PRAXIS, dimension)
        #opt = nlopt.opt(nlopt.LN_BOBYQA, dimension)
        opt = nlopt.opt(nlopt.LN_NELDERMEAD, dimension)
        #opt = nlopt.opt(nlopt.LN_COBYLA, dimension)
        #opt = nlopt.opt(nlopt.LD_MMA, dimension)

        lower_bounds = np.append(
            -1000*frequency[-1] * np.ones(nzero+npole), 1e-8)
        upper_bounds = np.append(
            -0.001*frequency[0] * np.ones(nzero+npole), dcgain_limit)

        #lower_bounds[-1] = 40

        opt.set_min_objective(error)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_ftol_rel(1e-6)
        vector = opt.optimize(np.append(-np.ones(nzero+npole),
                                        np.min([gain[0], upper_bounds[-1]])))
        print(vector)


        #opp = scipy.optimize.minimize(error, xo, method='SLSQP',
        #                              args=(frequency_matrix, gain))

        #print(opp)

        #plt.plot(frequency,
        #         system_gain / np.absolute(frequency_matrix @ denominator),
        #         '-C3')
        #plt.plot(frequency, gain, '.-')

        frequency = np.logspace(-2, 2, 50)
        plt.plot(frequency, model_gain(vector),
                 label=f'limit {dcgain_limit:1.0e}')
        #print(poles)

        return field_model(vector)

        #scipy.signal.lti()





if __name__ == '__main__':
    '''
    ftp = FTPData('')
    for experiment in ftp.listdir(select='CSJA')[2:]:
        for phase in Campaign(experiment).index:
            for side in ['Left', 'Right']:
                fluidresponse = FluidResponse(experiment, phase, side, [6], 0)
    '''

    fluid = FluidResponse('CSJA13', 0, 'Left')
    #response.read_data()
    frequency, gain = fluid.response(2)
    plt.plot(frequency, gain, 'o-', label='fluid gain')

    fluid.plot(2, dcgain_limit=1e8)
    #fluid.plot(2, dcgain_limit=5e2)
    #fluid.plot(2, dcgain_limit=2e2)

    plt.legend(loc='lower left')
    plt.despine()
    plt.xlabel('$\omega$ rads$^{-1}$')
    plt.ylabel('gain')
    plt.xscale('log')
    plt.yscale('log')

    '''
    omega, gain = response.response(2)
    index = -1

    omega = omega[index]
    cycles = 50
    t = np.linspace(0, cycles*2*np.pi/omega, 30*cycles)
    y = np.sin(omega*t)
    p = scipy.signal.lsim(lti, y, t)[1]

    print(2*np.mean(p**2), gain[index])

    plt.figure()
    plt.plot(t, y)
    plt.plot(t, p)
    #response.plot(9)

    #response = FluidResponse('CSJA11', -1, 'Left')
    #response.plot(2)
    #response.plot(9)
    #fluidresponse.sample.shot = 5
    #fluidresponse.fluidprofile.plot()
    '''

    """
    fluidresponse = FluidResponse('CFETR', 0, 'Left')
    index = fluidresponse.sample.trial.plan[('Be', 'T')] == 2
    coefficents = fluidresponse.load_data()
    plt.loglog(2*coefficents['frequency'][index],
               coefficents['steadystate'][index])

    fluidresponse = FluidResponse('CSJA_8', 0, 'Left')
    index = fluidresponse.sample.trial.plan[('Be', 'T')] == 2
    coefficents = fluidresponse.load_data()
    plt.loglog(2*coefficents['frequency'][index],
               coefficents['steadystate'][index])
    """
