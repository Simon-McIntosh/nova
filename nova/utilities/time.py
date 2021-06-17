import time
import datetime
import sys
import functools
import gc
import itertools

import numpy as np
from scipy.optimize import minimize
from timeit import default_timer as _timer

from nova.utilities.pyplot import plt


def timeit(_func=None, *, repeat=1, number=1000, file=sys.stdout):
    '''
    timeit decorator taken from:
        https://github.com/realpython/materials/blob/master/
        pandas-fast-flexible-intuitive/tutorial/timer.py
    '''
    """Decorator: prints time from best of `repeat` trials.
    Mimics `timeit.repeat()`, but avg. time is printed.
    Returns function result and prints time.
    You can decorate with or without parentheses, as in
    Python's @dataclass class decorator.
    kwargs are passed to `print()`.
    >>> @timeit
    ... def f():
    ...     return "-".join(str(n) for n in range(100))
    ...
    >>> @timeit(number=100000)
    ... def g():
    ...     return "-".join(str(n) for n in range(10))
    ...
    """

    _repeat = functools.partial(itertools.repeat, None)

    def wrap(func):
        @functools.wraps(func)
        def _timeit(*args, **kwargs):
            # Temporarily turn off garbage collection during the timing.
            # Makes independent timings more comparable.
            # If it was originally enabled, switch it back on afterwards.
            gcold = gc.isenabled()
            gc.disable()

            try:
                # Outer loop - the number of repeats.
                trials = []
                for _ in _repeat(repeat):
                    # Inner loop - the number of calls within each repeat.
                    total = 0
                    for _ in _repeat(number):
                        start = _timer()
                        result = func(*args, **kwargs)
                        end = _timer()
                        total += end - start
                    trials.append(total)

                # We want the *average time* from the *best* trial.
                # For more on this methodology, see the docs for
                # Python's `timeit` module.
                #
                # "In a typical case, the lowest value gives a lower bound
                # for how fast your machine can run the given code snippet;
                # higher values in the result vector are typically not
                # caused by variability in Python’s speed, but by other
                # processes interfering with your timing accuracy."
                best = min(trials) / number
                print(
                    "Best of {} trials with {} function"
                    " calls per trial:".format(repeat, number)
                )
                print(
                    "Function `{}` ran in average"
                    " of {:0.3f} seconds.".format(func.__name__, best),
                    end="\n\n",
                    file=file,
                )
            finally:
                if gcold:
                    gc.enable()
            # Result is returned *only once*
            return result

        return _timeit

    # Syntax trick from Python @dataclass
    if _func is None:
        return wrap
    else:
        return wrap(_func)


class clock(object):

    def __init__(self, nITER=0, print_rate=1, print_width=22, header=''):
        self.start(nITER, header)
        self.rate = print_rate
        self.width = print_width

    def start(self, nITER, header):
        self.i = 0
        self.to = time.time()
        self.nITER = nITER
        if self.nITER > 0:
            self.nint = int(np.log10(self.nITER))+1
        else:
            self.nint = 0
        self.time = True
        if header:
            print(header)

    def write(self, txt):
        sys.stdout.write(txt)
        sys.stdout.flush()

    def tock(self):
        self.i += 1
        if self.i % self.rate == 0 and self.i <= self.nITER:
            elapsed = time.time() - self.to
            remain = int((self.nITER - self.i) / self.i * elapsed)
            txt = f'\r{{:{self.nint}d}}'
            txt = txt.format(self.i)
            txt += ' elapsed {:0>8}s'.format(str(
                    datetime.timedelta(seconds=int(elapsed))))
            txt += ' remain {:0>8}s'.format(str(
                    datetime.timedelta(seconds=remain)))
            txt += ' complete {:5.1f}%'.format(
                    1e2 * self.i / self.nITER)
            nh = int(self.i / self.nITER * self.width)
            txt += ' |' + nh * '#' + (self.width - nh) * '-' + '|'
            self.write(txt)
        if self.i == self.nITER:
            self.stop()

    def stop(self):
        if self.time:
            elapsed = time.time() - self.to
            txt = f'{{:{55+self.nint+self.width}}}'  # flush
            txt = txt.format(f'\rtotal elapsed time {elapsed:1.4f}s')
            self.write(txt)
            self.write('\n')
            self.time = False


class time_constant:

    def __init__(self, td, Id, trim_fraction=0.2):
        self.load(td, Id)
        self.trim_fraction = trim_fraction

    def load(self, td, Id):  # load profile
        self.td = np.copy(td)
        self.Id = np.copy(Id)

    def trim(self, **kwargs):
        self.trim_fraction = kwargs.get('trim_fraction', self.trim_fraction)
        if self.trim_fraction > 0 and self.trim_fraction < 1\
                and self.Id[0] != 0:
            dI = self.Id[-1] - self.Id[0]
            i1 = next((i for i, Id in enumerate(self.Id)
                       if abs(Id) < abs(self.Id[0] +
                                        (1-self.trim_fraction)*dI)))
            td, Id = self.td[:i1], self.Id[:i1]
        else:
            td, Id = self.td, self.Id
        return td, Id

    def fit_tau(self, x, *args):
        to, Io, tau = x
        t_exp, I_exp = args
        I_fit = Io*np.exp(-(t_exp-to)/tau)
        err = np.sum((I_exp-I_fit)**2)  # sum of squares
        return err

    def get_tau(self, plot=False, **kwargs):
        td, Id = self.trim(**kwargs)
        to = kwargs.get('to', 10e-3)
        Io = kwargs.get('Io', -60e3)
        tau = kwargs.get('tau', 30e-3)
        x = minimize(self.fit_tau, [to, Io, tau], args=(td, Id)).x
        err = self.fit_tau(x, td, Id)
        to, Io, tau = x
        Iexp = Io*np.exp(-(td-to)/tau)
        if plot:
            ax = kwargs.get('ax', plt.gca())
            ax.plot(1e3*td, 1e-3*Iexp, '-', label='exp')
        return to, Io, tau, err, Iexp

    def get_td(self, plot=False, **kwargs):  # linear discharge time
        td, Id = self.trim(**kwargs)
        A = np.ones((len(td), 2))
        A[:, 1] = td
        a, b = np.linalg.lstsq(A, Id, rcond=None)[0]
        tlin = abs(self.Id[0]/b)  # discharge time
        Ilin = a + b*td  # linear fit
        err = np.sum((Id-Ilin)**2)
        if plot:
            plt.plot(1e3*td, 1e-3*Ilin, '-', label='lin')
        return a, b, tlin, err, Ilin

    def fit_ntau(self, x, *args):
        texp, Iexp = args
        Ifit = self.I_nfit(texp, x)
        err = np.sum((Iexp-Ifit)**2)  # sum of squares
        return err

    def I_nfit(self, t, x):
        n = int(len(x)/2)
        Io, tau = x[:n], x[n:]
        Ifit = np.zeros(len(t))
        for Io, tau in zip(x[:n], x[n:]):
            Ifit += Io*np.exp(-t/tau)
        return Ifit

    def nfit(self, n, plot=False, **kwargs):
        tau_o = kwargs.get('tau_o', 50e-3)  # inital timeconstant
        td, Id = self.trim(**kwargs)
        to = td[0]
        td -= to  # time shift
        xo = np.append(Id[0]/n*np.ones(n),
                       np.sort(tau_o*np.ones(n) * np.random.random(n))[::-1])
        #xo *= np.random.random(2*n)
        bounds = [(None, None) for __ in range(n)]
        bounds.extend([(1e-6, None) for __ in range(n)])
        x = minimize(self.fit_ntau, xo, args=(td, Id), tol=1e-5).x
        #             method='L-BFGS-B', bounds=bounds).x
        Ifit = self.I_nfit(td, x)
        Io, tau = x[:n], x[n:]
        if plot:
            if 'ax' in kwargs:
                ax = kwargs['ax']
            else:
                ax = plt.subplots(1, 1)[1]
            ax.plot(1e3*td+to, 1e-3*Id, '-', label='data')
            ax.plot(1e3*td+to, 1e-3*Ifit, '--', label='fit')
            ax.set_xlabel('$t$ ms')
            ax.set_ylabel('$I$ kA')
            plt.despine()
            plt.legend()
        return Io, tau, td+to, Ifit

    def ntxt(If, tau):
        txt = r'$\alpha$=['
        for i, I in enumerate(If):
            if i > 0:
                txt += ','
            txt += '{:1.2f}'.format(I)
        txt += ']'
        txt += r' $\tau$=['
        for i, t in enumerate(tau):
            if i > 0:
                txt += ','
            txt += '{:1.1f}'.format(1e3*t)
        txt += ']ms'
        return txt

    def fit(self, plot=False, **kwargs):
        tfit, Idata = self.trim(**kwargs)
        tau, tau_err, Iexp = self.get_tau(plot=False, **kwargs)[-3:]
        tlin, tlin_err, Ilin = self.get_td(plot=False, **kwargs)[-3:]
        if tau_err < tlin_err:
            self.discharge_type = 'exponential'
            self.discharge_time = tau
            Ifit = Iexp
        else:
            self.discharge_type = 'linear'
            self.discharge_time = tlin
            Ifit = Ilin
        if plot:
            if 'ax' in kwargs:
                ax = kwargs['ax']
            else:
                ax = plt.subplots(1, 1)[1]

            ax.plot(1e3*tfit, 1e-3*Idata, '-', label='data')
            ax.plot(1e3*tfit, 1e-3*Ifit, '--',
                    label='{} fit'.format(self.discharge_type))
            ax.set_xlabel('$t$ ms')
            ax.set_ylabel('$I$ kA')
            plt.despine()
            # plt.legend()
            txt = '{} discharge'.format(self.discharge_type)
            txt += ', t={:1.1f}ms'.format(self.discharge_time)
            # plt.title(txt)
        return self.discharge_time, self.discharge_type, tfit, Ifit
