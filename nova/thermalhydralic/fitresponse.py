# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:38:22 2021

@author: mcintos
"""


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
        frequency = self.coefficents['frequency'][pareto_index].to_numpy()
        gain = self.coefficents['steadystate'][pareto_index].to_numpy()
        return frequency, gain**1.6

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
            return 10**(dBgain / 20)

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
        opt = nlopt.opt(nlopt.LN_BOBYQA, dimension)
        #opt = nlopt.opt(nlopt.LN_NELDERMEAD, dimension)
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
        opt.set_initial_step(0.001)
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



    '''
    omega, gain = response.response(2)
    index = -1

    omega = omega[index]
    cycles = 50
    t = np.linspace(0, cycles*2*np.pi/omega, 30*cycles)
    y = np.sin(omega*t)
    p = scipy.signal.lsim(lti, y, t)[1]

    print(2*np.mean(p**2), gain[index])
    '''