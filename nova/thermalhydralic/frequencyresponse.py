"""Frequency response post-processing for AC Loss data."""

import pandas
import numpy as np

from nova.thermalhydralic.plotimport import pyplot


class FrequencyResponse:
    """Fit magnitude response and extract system transfer function."""

    def __init__(self):
        self._data = {}

    @property
    def data(self):
        """Return the appended datasets keyed by label, read-only."""
        return self._data

    def append_data(self, input_frequency, rms_power, label=None, prefix="dataset"):
        """
        Append input data.

        Parameters
        ----------
        input_frequency : array-like
            Dataset input frequency, rad/s.
        rms_power : array-like
            rms power output.
        label : str, optional
            Dataset label. The default is None.
        prefix : str, optional
            Dataset prefix, used if label=None. The default is 'dataset'.

        Raises
        ------
        IndexError
            Duplicate label found in input dataset.

        Returns
        -------
        None.

        """
        # the two inputs are columns of one frame, one row per sample: passing
        # them as a row list instead builds a two-row frame and only parses at
        # all when the sweep happens to hold exactly two frequencies
        data = pandas.DataFrame({"frequency": input_frequency, "rms_power": rms_power})
        data.sort_values(["frequency"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        data["magnitude"] = 20 * np.log10(data["rms_power"])
        if label is None:
            offset = sum(1 for name in self._data if prefix in name)
            label = f"{prefix}{offset}"
        if label in self._data:
            raise IndexError(
                f"label {label} already present in dataset {list(self._data)}"
            )
        self._data[label] = data

    def plot_data(self, ax=None):
        """Plot the magnitude response of every appended dataset."""
        if ax is None:
            ax = pyplot().gca()
        for label in self._data:
            ax.plot(self._data[label]["frequency"], self._data[label]["magnitude"], "o")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\omega$ rads$^{-1}$")
        ax.set_ylabel(r"$20\log_{10}|H|$ dB")


if __name__ == "__main__":
    """

    def _H(x):
        H = x[0]
        for bp in x[1:]:
            H /= (omega+bp)
        return H

    def bode(x):
        zeros, poles, gain = [], -x[:-1], x[-1]
        magnitude = scipy.signal.bode((zeros, poles, gain), w=omega)[1]
        H = 10**(mag/20)
        return H

    def bode_err(x):
        H = bode(x)
        return np.sqrt(np.mean((np.log10(p) - np.log10(H))**2))

    def log_rms(x, grad):
        err = bode_err(x)
        print(err)
        if len(grad) > 0:
            grad[:] = scipy.optimize.approx_fprime(x, bode_err, 1e-6)
        return err


    xo = [6, 10, 10]
    opt = nlopt.opt(nlopt.LD_MMA, len(xo))
    opt.set_min_objective(log_rms)
    opt.set_ftol_rel(1e-6)
    #opt.set_xtol_rel(self.xtol_rel)
    x = opt.optimize(xo)
    print(x, opt.last_optimize_result())

    #opt.set_ftol_rel(self.ftol_rel)
    #opt.set_xtol_rel(self.xtol_rel)
    #opt.set_lower_bounds(self.grid_boundary[::2])
    #opt.set_upper_bounds(self.grid_boundary[1::2])



    xo = [0.1, 2, 20]
    bounds = [(np.min(omega), np.max(omega)) for __ in range(len(xo))]
    bounds[-1] = (1e-6, None)
    res = scipy.optimize.minimize(bode_err, xo, method='L-BFGS-B',
                                  bounds=bounds, options={'gtol': 1e-9})
    tau = np.sort(2*np.pi/res.x[:-1])
    """
