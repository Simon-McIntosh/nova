"""Post-process CS module cold-test instrumentation logs.

Reads the per-channel CSV exports of a coil cold test, groups the channels by
sensor type and unit, conditions them onto a common timebase and fits the
current-squared dependence that qualifies the coil.
"""

import datetime
import os
from glob import glob
import string

import numpy as np
import pandas
from scipy.interpolate import interp1d
import scipy.signal

from nova.definitions import root_dir
from nova.thermalhydralic.plotimport import clock, gridspec, pyplot, seaborn
from nova.utilities.IO import pythonIO


def _linelabel(*args, **kwargs):
    """Return a line-end label collector.

    The collector draws through pylab, so it is reached from plot methods only
    and never at module scope.
    """
    from nova.graphics.addtext import linelabel

    return linelabel(*args, **kwargs)


class cold_test(pythonIO):
    _ID = {
        "TCS": "TCS",
        "DS": "displace",
        "SG": "strain",
        "TC": "TCS",
        "EX": "extend",
        "ST": "strain",
    }
    _units = {
        "%Open": "valve",
        "kA": "current",
        "K": "temperature",
        "mOhm": "resistance",
        "V": "voltage",
        "mV": "mvoltage",
        "W": "power",
        "bara": "apressure",
        "mbar": "pressure",
        "g/s": "flow",
        "ppm": "strain",
        "mm": "extend",
    }
    _labels = {
        "current": "$I$ kA",
        "temperature": "$T$ K",
        "strain": r"$\epsilon$ ppm",
        "extend": r"$\Delta W$ mm",
        "displace": r"$\Delta W$ mm",
    }

    def __init__(self, project_dir="CSM2", read_txt=False):
        super().__init__()  # python read/write
        self.directory = os.path.join(root_dir, "data/CSM", project_dir)
        self.module = project_dir
        self.groups, self.channels = [], {}
        self._units_r = {self._units[u]: u for u in self._units}
        self.to = np.datetime64("2020-01-07T08:33:00.000000000")
        self.read_txt = read_txt

    def split_channels(self, data):
        index = {
            **{self._ID[prefix]: [] for prefix in self._ID},
            **{self._units[unit]: [] for unit in self._units},
            **{"other": []},
        }
        for channel in data.columns:
            if np.array([prefix in channel[0] for prefix in self._ID]).any():
                for label in self._ID:
                    if label in channel[0]:
                        index[self._ID[label]].append(channel)
                        break
            elif channel[1] in self._units:
                index[self._units[channel[1]]].append(channel)
            else:
                index["other"].append(channel)
        for channel in list(index.keys()):  # remove empty channels
            if len(index[channel]) == 0:
                index.pop(channel)
        return index

    def load_coldtest(self, group, **kwargs):
        read_txt = kwargs.get("read_txt", self.read_txt)
        filepath = os.path.join(self.directory, group)
        rawfilepath = os.path.join(self.directory, "rawdata")
        if read_txt or not os.path.isfile(filepath + ".pk"):
            if group in ["strain", "extend"]:
                data = self.read_coldtest(group, "strain")
            else:
                if group != "rawdata" and os.path.isfile(rawfilepath + ".pk"):
                    if not hasattr(self, "rawdata"):
                        self.load_pickle(rawfilepath)  # load rawdata
                    data = self.read_rawdata(group)
                else:
                    # cooldown = self.read_coldtest(group, 'cooldown')
                    data = self.read_coldtest(group, "test")
                    # data = pandas.concat((cooldown, test), sort=True)
                    data.sort_index(inplace=True)
            if group != "rawdata":
                self.condition_signal(data)
            setattr(self, group, data)
            self.save_pickle(filepath, [group])
        else:
            self.load_pickle(filepath)
        self.groups.append(group)
        self.channels.update(
            **{channel: group for channel in getattr(self, group).columns.droplevel(1)}
        )

    def read_folder(self, *args):
        files = glob(os.path.join(self.directory, *args))
        files = np.sort([f.split("\\")[-1] for f in files])
        return files

    def read_rawdata(self, group):
        channel_index = self.split_channels(self.rawdata)
        if group == "resistance":
            self.rawdata[("RCoil", "mOhm")].fillna(
                1e-3 * self.rawdata[("RCoil", "ohm")], inplace=True
            )
        data = self.rawdata.loc[:, channel_index[group]]
        if group == "resistance":
            data.loc[:] *= 1e3  # all to mOhm
        return data

    def read_coldtest(self, group, folder, to=None):
        if to is None:
            to = self.to
        files = self.read_folder(folder, "*.csv")
        nfiles = len(files)
        data = [[] for __ in range(nfiles)]
        tick = clock(nITER=nfiles, header=f"reading {nfiles} {folder} files")
        if group in ["strain", "extend"]:
            folder = "strain"
        for i, file in enumerate(files):
            if folder == "strain":
                file_reader = self.read_strain_file
            else:
                file_reader = self.read_coldtest_file
            data[i] = file_reader(folder, file)
            channel_index = self.split_channels(data[i])
            if group != "rawdata":
                data[i] = data[i].loc[:, channel_index[group]]
            tick.tock()
        if data:
            data = pandas.concat(data, sort=True)
            data.sort_index(inplace=True)
            data = data[~data.index.duplicated(keep="first")]
        return data

    def read_coldtest_file(self, folder, filename):
        data = pandas.read_csv(os.path.join(self.directory, folder, filename), header=7)
        data["timestamp"] = pandas.to_datetime(data.timestamp)
        self.format_columns(data)
        if "20200107" in filename:  # trim start of file
            data = data.iloc[28:, :]
        return data

    def read_strain_file(self, folder, filename):
        filepath = os.path.join(self.directory, folder, filename)
        data = pandas.read_csv(filepath, header=7)
        if self.module == "CSM1":
            data = data.iloc[:, 3:]  # drop first three rows
            elapsed = pandas.read_csv(filepath, header=7, usecols=[0])
        else:
            data = data.iloc[:, 1:]  # drop first rows
            elapsed = pandas.read_csv(filepath, header=7, usecols=[0])
        print(data)
        columns = {}
        for c in data.columns:
            if c[:2] == "EX":
                columns[c] = f"{c.split(' on ')[-1]}_(mm)"
            elif c[:2] == "SG":
                columns[c] = f"{c}_(ppm)"
            else:
                columns[c] = c
        data.rename(columns=columns, inplace=True)
        data["timestamp"] = pandas.to_datetime(elapsed.iloc[:, 0])
        if "021320 - 021420" in filename:  # correct time shift
            data.loc[:, "timestamp"] -= datetime.timedelta(seconds=21 * 60 + 21)
        self.format_columns(data)
        if self.module == "CSM2":
            # five gauge vertical strain
            self.mean_strain("STvID", data, range(119, 124))
            self.mean_strain("STvOD", data, range(124, 129))
            # three gauge hoop strain, ID
            self.mean_strain("SThID0", data, [101, 107])
            self.mean_strain("SThID1", data, [108, 114])
            self.mean_strain("SThID2", data, [103, 109, 115])
            # three gauge hoop strain, OD
            self.mean_strain("SThOD0", data, [104, 110, 116])
            self.mean_strain("SThOD1", data, [105, 111, 117])
            self.mean_strain("SThOD2", data, [106, 112, 118])
        # rename ppm
        data.rename({"uStrain": "ppm"}, level=1, axis=1, inplace=True)
        return data

    def mean_strain(self, label, data, index):
        """Calculate mean strain."""
        columns = [col for i in index if (col := f"ST{i}") in data]
        data.loc[:, (label, "uStrain")] = data.loc[:, columns].mean(1)

    def format_columns(self, data):
        data.rename(columns={"timestamp": "timestamp_(time}"}, inplace=True)
        columns = data.columns.values
        _columns = [c.replace("mean", "").replace("_", "") for c in columns]
        _columns = [c.replace("value", "") for c in _columns]
        labels = [c.split("(")[0] for c in _columns]
        units = [c.split("(")[1][:-1] for c in _columns]
        data.rename(
            columns=dict(zip(columns, labels)),
            inplace=True,
        )
        data.columns = pandas.MultiIndex.from_arrays(
            [data.columns, units], names=("ID", "unit")
        )
        data.set_index(("timestamp", "time"), inplace=True)
        data.index.name = "timestamp"
        # data.rename(columns={'Ibusfast': 'IBus', 'PSIOutfast': 'PSIOut'},
        #            inplace=True, level=0)

    def t(self, index, to=None):
        """Return seconds elapsed from to, as a float array.

        Every consumer differentiates or interpolates against this, so it has to
        be a plain float array: casting the difference to a millisecond
        resolution leaves a timedelta index, and scaling that yields timedelta
        objects rather than the seconds the callers expect.
        """
        if to is None:
            to = self.to
        elif isinstance(to, int):
            to = index[to]
        return np.asarray((index - to) / np.timedelta64(1, "s"), dtype=float)

    def condition_signal(self, data):
        t = self.t(data.index)
        Tcol = [c for c in data.columns if c[0][:2] == "TT"]
        DScol = [c for c in data.columns if c[0][:2] in ["DS", "EX"]]
        strain_col = [c for c in data.columns if c[0][:2] == "SG"]
        current_col = [c for c in data.columns if c[1] == "A" or c[1] == "kA"]
        for col in data.columns:
            if col in Tcol:
                signal_minimum = data.loc[:, col] < 1e-3
                signal_maximum = data.loc[:, col] > 300
                gradient_max = 0.1
            elif col in DScol:
                signal_minimum = data.loc[:, col] < -30
                signal_maximum = data.loc[:, col] > -1e-4
                gradient_max = 0.1
            elif col in current_col:
                if col[0] == "IBus":
                    data.loc[data.loc[:, col] > 45, col] *= 1e-3
                if col[1] == "kA":
                    signal_minimum = data.loc[:, col] < -45
                    signal_maximum = data.loc[:, col] > 45
                    gradient_max = 0.1
                else:
                    signal_minimum = False
                    signal_maximum = False
                    gradient_max = False  # 0.01
            elif col in strain_col:
                data.loc[:, col] *= 2
                signal_minimum = data.loc[:, col] < -5000
                signal_maximum = data.loc[:, col] > 5000
                gradient_max = False
            elif col[0] == "RCoil":
                signal_minimum = data.loc[:, col] < 0
                signal_maximum = False
                gradient_max = 0.1
            else:
                signal_minimum = False
                signal_maximum = False
                gradient_max = False
            if gradient_max:
                value = data.loc[:, col].copy()
                isna = value.isna()
                value.loc[isna] = 0
                # unique returns positions into t, and the channel is indexed by
                # timestamp, so the samples it picks out must be read positionally
                _t, _i = np.unique(t, return_index=True)
                _v = value.to_numpy()[_i]
                gradient = interp1d(_t, np.gradient(_v, _t))(t)
                gradient_maximum = (abs(gradient) > gradient_max) | isna
            else:
                gradient_maximum = np.zeros(data.shape[0], dtype=bool)
            index = signal_minimum | signal_maximum | gradient_maximum
            # interpolate
            if sum(~index) > 2:  # surficent data for interpolation
                interp = interp1d(
                    t[~index],
                    data.loc[~index, col],
                    bounds_error=False,
                    fill_value=None,
                )
                data.loc[index, col] = interp(t[index])
                # calculate high frequency content
                dt = np.median(np.diff(t))
                # welch slices its input with an ellipsis, which a labelled
                # series does not accept
                __, psd = scipy.signal.welch(data.loc[:, col].to_numpy(), 1 / dt)
                if psd[-1] > 1 and col in Tcol:
                    data.drop(columns=col, inplace=True)
            elif col in Tcol:
                data.drop(columns=col, inplace=True)
        dt = np.append(t[1:] - t[:-1], 0)
        gap = np.median(dt) + 1.5 * np.std(dt)
        data.loc[dt > gap, :] = None

    def plot_temperature(self):
        """Plot every temperature channel, coloured by its final value."""
        plt = pyplot()
        ax = plt.subplots(1, 1)[1]
        text = _linelabel(Ndiv=50, value="")
        for col in self.temperature:
            T = self.temperature.loc[:, col]
            # the channel is indexed by timestamp, so the last sample is
            # positional; -1 is not a label in it
            final = T.iloc[-1]
            if final > 250:
                color = "C0"
            elif final > 50:
                color = "C1"
            elif final > 10:
                color = "C2"
            elif final > 6:
                color = "C4"
            else:
                color = "C3"
            ax.plot(self.temperature.loc[:, col], T, color=color, label=col[0][2:])
            text.add(col[0])
        ax.set_ylim([0, 300])
        ax.set_xlabel("$t$ hr")
        ax.set_ylabel("$T$ K")
        ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.4))
        seaborn().despine(ax=ax)

    def offset(self, dataframe, offset_dt):
        index = [[], []]
        index[0] = dataframe.index[0]
        index[1] = index[0] + datetime.timedelta(seconds=offset_dt)
        offset = dataframe.loc[index[0] : index[1]].mean()
        return offset

    def get_dataframe(self, label, index):
        index = self.get_index(index)
        if isinstance(label, pandas.DataFrame):
            dataframe = label
            group = self.channels[dataframe.columns.droplevel(1)[0]]
        else:
            if pandas.api.types.is_list_like(label):
                group = self._ID[label[0][:2]]
                self.load_coldtest(group)
                groups = [self.channels[name] for name in label]
                if not np.array([g == groups[0] for g in groups]).all():
                    raise IndexError("list labels must belong to same group")
                group = groups[0]
                channels = [(name, self._units_r.get(group, "mm")) for name in label]
            elif label in self.groups:
                group = label
                channels = slice(None)
            elif label in self.channels:
                group = self.channels[label]
                channels = [(label, self._units_r[group])]
            else:
                group = label
                channels = slice(None)
            if not hasattr(self, group):
                self.load_coldtest(group)  # load dataframe
            dataframe = getattr(self, group).copy()  # get dataframe
            if group in ["current"]:
                dataframe.drop(columns=["IBus"], inplace=True)
                # dataframe.drop(columns=['PSIOut'], inplace=True)
                channels = dataframe.columns[::-1]
            dataframe = dataframe.loc[:, channels]
        dataframe = dataframe[index]
        return dataframe, group

    def plot(
        self,
        label,
        index=None,
        ax=None,
        offset=None,
        offset_dt=5 * 60,
        legend=True,
        labels=True,
        xlabel=True,
        ylabel=True,
        ncol=2,
        **kwargs,
    ):
        plt = pyplot()
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()
        dataframe, group = self.get_dataframe(label, index)
        if group in ["current", "temperature"]:
            offset_dt = 0
        # self.condition_signal(dataframe)
        if offset is not None:
            dataframe -= offset
        elif offset_dt > 0:  # calculate offset
            offset = self.offset(dataframe, offset_dt)
            dataframe -= offset
        else:
            offset = 0
        color = kwargs.get("color", None)
        for i, col in enumerate(dataframe):
            if color is None:
                kwargs["color"] = self.get_color(i, col)
            ax.plot(dataframe[col], label=col, **kwargs)
        if legend and labels:
            shift = np.floor(dataframe.shape[1] / ncol) * 0.12
            ax.legend(
                [c for c in dataframe.columns.droplevel(1)],
                ncol=ncol,
                loc="upper center",
                bbox_to_anchor=(0.5, 1 + shift),
            )
        seaborn().despine(ax=ax)
        fig.autofmt_xdate()
        if labels:
            if xlabel:
                ax.set_xlabel("timestamp")
            if group in self._labels and ylabel:
                ax.set_ylabel(self._labels[group])
        return offset

    def plot_col(self, label, index=["cooldown", "test"], offset_dt=5 * 60):
        """Plot the cooldown and test phases side by side on a shared scale."""
        ax = pyplot().subplots(1, 2, sharey=True, gridspec_kw={"wspace": 0.1})[1]
        offset = self.plot(label, index=index[0], ax=ax[0], offset_dt=offset_dt)
        self.plot(label, index=index[1], ax=ax[1], labels=False, offset=offset)

    def plot_row(self, label, index="test", ncol=2, color=None, offset_dt=5 * 60):
        """Plot a channel group above the drive current on a shared timebase."""
        plt = pyplot()
        fig = plt.figure()
        gs = gridspec().GridSpec(2, 1, height_ratios=[6, 1])
        ax = []
        ax.append(fig.add_subplot(gs[0]))
        ax.append(fig.add_subplot(gs[1], sharex=ax[0]))
        plt.setp(ax[0].get_xticklabels(), visible=False)
        legend = ncol > 0
        self.plot(
            label,
            index=index,
            ax=ax[0],
            xlabel=False,
            ncol=ncol,
            color=color,
            legend=legend,
            offset_dt=offset_dt,
        )
        self.plot("current", index=index, ax=ax[1], legend=False, color="gray")
        Imax_iloc = np.nanargmax(self.current["PSIOut"])
        self.current.iloc[Imax_iloc]["PSIOut"]
        # ax[1].text(, Imax, f'{Imax}')
        return ax

    def extract_shrinkage(self, sensors=["displace", "extend"], plot=True, Imax=48.5):
        for group in sensors + ["current"]:
            if group not in self.groups:
                self.load_coldtest(group)
        shrink = pandas.DataFrame(index=self.current.index)
        if "extend" in sensors:
            self.condition_signal(self.extend)
            self.extend.loc[slice("2020-02-13 12:00", "2020-02-14"), "EX-170-ID"] = None
            t = self.t(shrink.index)  # timebase
            t_ex = self.t(self.extend.index)  # strain timebase
            for col in self.extend:  # load extension results
                shrink[col] = interp1d(
                    t_ex, self.extend[col], bounds_error=False, fill_value=None
                )(t)
        if "displace" in sensors:
            for col in self.displace:  # load displacment results
                shrink[col] = self.displace[col]
        # clip
        for col in shrink:
            index = shrink[col] < -9
            shrink.loc[index, col] = None
        shrink.columns = pandas.MultiIndex.from_tuples(shrink.columns)
        setattr(self, "shrink", shrink)
        self.groups.append("shrink")
        self.fit(shrink, index="test", plot=plot, Imax=Imax)

    def get_current(self, label, index):
        index = self.get_index(index)
        # extract current
        if not hasattr(self, "current"):
            self.load_coldtest("current")
        current = self.current.loc[index, ("PSIOut", "kA")]
        # current = self.current.loc[index, ('IBus', 'kA')]
        # extract dataframe
        dataframe, group = self.get_dataframe(label, index)
        # interpolate
        if current.shape[0] == dataframe.shape[0]:
            interpolate = not np.equal(current.index, dataframe.index).all()
        if current.shape[0] != dataframe.shape[0] or interpolate:
            t = self.t(current.index)  # current time index
            t_data = self.t(dataframe.index)  # data time index
            _dataframe = pandas.DataFrame(index=current.index)
            for col in dataframe:  # load extension results
                _dataframe[col] = interp1d(
                    t_data, dataframe.loc[:, col], bounds_error=False, fill_value=None
                )(t)
            dataframe = _dataframe
            dataframe.columns = pandas.MultiIndex.from_tuples(dataframe.columns)
        # zero offset
        dataframe -= self.offset(dataframe, 5)
        return current, dataframe

    def get_color(self, i, col):
        """Return line color."""
        try:
            index = int(col[0].lstrip(string.ascii_letters))
            if col[0][:2] == "ST":
                index += 1
        except ValueError:
            index = i + 1
        return f"C{index - 1}"

    def plot_loop(self, label, index="CSM2", ncol=2):
        """Plot each channel against drive current over the ramp loop."""
        current, dataframe = self.get_current(label, index)
        group = self.channels[dataframe.columns.droplevel(1)[0]]
        ax = pyplot().subplots(1, 1)[1]
        max_value = dataframe.abs().max().max()
        for i, col in enumerate(dataframe):
            if (
                not np.isnan(dataframe.loc[:, col]).all()
                and dataframe.loc[:, col].abs().max() > 0.05 * max_value
            ):
                value = scipy.signal.savgol_filter(dataframe[col], 51, 1)
                color = self.get_color(i, col)
                ax.plot(current, value, "-", label=col[0], color=color, lw=1.5)
        seaborn().despine(ax=ax)
        ax.set_xlabel("$I$ kA")
        ax.set_ylabel(self._labels[group])
        shift = np.floor(dataframe.shape[1] / ncol) * 0.12
        ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, 1 + shift))

    def fit(
        self,
        label,
        index="test",
        Imin=7.5,
        Itrim=40,
        Imax=40,
        plot=True,
        ncol=2,
        color=None,
        ax=None,
        trim=True,
        loc=None,
        Ndiv=20,
        xlabel=True,
    ):
        """Return the current-squared coefficient of each selected channel.

        A cold-test channel that responds to Lorentz loading grows with the
        square of the drive current, so one coefficient per channel is the
        qualification result. The fit uses the down-ramp between Imin and Itrim,
        where the coil is in equilibrium with the load rather than chasing it.
        """
        current, dataframe = self.get_current(label, index)
        current_rate = np.gradient(current, self.t(current.index))
        # trim  current
        if trim:
            current_index = (
                (current >= Imin) & (current <= Itrim) & (current_rate < -0.01)
            )
            current = current[current_index]
            dataframe = dataframe[current_index]
        max_value = dataframe.abs().max().max()
        coef = np.zeros(dataframe.shape[1])
        for i, col in enumerate(dataframe):
            if not np.isnan(dataframe.loc[:, col]).all():
                index = dataframe.loc[:, col].notna()
                coef[i] = np.linalg.lstsq(
                    current[index].to_numpy().reshape(-1, 1) ** 2,
                    dataframe.loc[index, col],
                    rcond=None,
                )[0][0]
        if plot:
            if ax is None:
                ax = pyplot().subplots(1, 1)[1]
            if Imax is None:
                Imax = current.max()
                xtick = None
            else:
                xtick = Imax
            current_fit = np.linspace(0, Imax, 100)  # plot fit
            group = self.channels[dataframe.columns.droplevel(1)[0]]
            if group == "strain":
                value = "1.0f"
            else:
                value = "1.2f"
            text = _linelabel(
                value=value, postfix=self._units_r.get(group, "mm"), Ndiv=Ndiv, ax=ax
            )
            for i, col in enumerate(dataframe):
                if dataframe[col].abs().max() < 0.05 * max_value:
                    continue
                if color is None:
                    c = self.get_color(i, col)
                else:
                    c = color
                if not np.isnan(dataframe.loc[:, col]).all():
                    ax.plot(
                        current,
                        dataframe[col],
                        ".",
                        color=c,
                        alpha=0.75,
                        ms=4,
                        zorder=-20,
                    )
                    ax.plot(
                        current_fit,
                        coef[i] * current_fit**2,
                        "--",
                        color=c,
                        alpha=1,
                        label=col[0],
                    )
                    text.add("")
            if xtick is not None:
                xticks = ax.get_xticks()
                dx = xticks[-1] - xticks[0]
                xticks = [x for x in xticks if abs(x - xtick) / dx > 0.1]
                xticks = np.sort(np.append(xticks, xtick))
                ax.set_xticks(xticks)
            seaborn().despine(ax=ax)
            if xlabel:
                ax.set_xlabel("$I$ kA")
            ax.set_ylabel(self._labels[group])
            shift = np.floor(dataframe.shape[1] / ncol) * 0.12
            if loc is None:
                ax.legend(
                    ncol=ncol,
                    loc="upper center",
                    bbox_transform=ax.transAxes,
                    bbox_to_anchor=(0.5, 1 + shift),
                )
            else:
                ax.legend(ncol=ncol, loc=loc)
            text.plot()
        return coef

    def get_index(self, index):
        if index == "cooldown":
            return slice("2020-01-07", "2020-01-27")  # cooldown
        if index == "test":
            return slice("2020-02-11 9:00", "2020-02-14 18:00")
        if index == "strain":
            return slice("2020-02-11 9:00", "2020-02-14 12:00")
        if index == "low_strain":
            return slice("2020-02-11 9:00", "2020-02-11 18:00")
        if index == "low_strain_trim":
            return slice("2020-02-11 9:00", "2020-02-11 12:40")
        if index == "medium_strain":
            return slice("2020-02-12 9:00", "2020-02-12 18:00")
        if index == "medium_strain_trim":
            return slice("2020-02-12 9:00", "2020-02-12 13:50")
        if index == "high_strain_trim":
            return slice("2020-02-14 9:30", "2020-02-14 11:10")
        if index == "fit":
            return slice("2020-02-11 9:00", "2020-02-12 18:00")
        if index == "drop":
            return slice("2020-02-12 16:00", "2020-02-12 17:00")
        if index == "drop_trim":
            return slice("2020-02-12 16:10", "2020-02-12 16:27:30")
        if index == "CSM1":
            return slice("2020-02-14", "2020-02-14")
        if index == "CSM2":
            return slice("2021-03-18", "2021-04-22")
        if index == "CSM2_trim":
            return slice("2021-04-08", "2021-04-09")
        if index == "CSM2_current":
            return slice("2021-04-08", "2021-04-09  13:20:00")
        if index == "CSM2_08":
            return slice("2021-04-08 13", "2021-04-08  16")
        if index == "CSM2_09":
            return slice("2021-04-09", "2021-04-09 13:20:00")
        if index == "CSM2_09_trim":
            return slice("2021-04-09 10:30:00", "2021-04-09 13:20:00")
        if index == "CSM2_19":
            return slice("2021-03-19", "2021-03-19")
        if index == "CSM4_a":
            return slice("2022-12-12 8:00", "2022-12-12 18:00")
        if index == "CSM4_b":
            return slice("2023-01-12", "2023-01-14")
        if index == "CSM4":
            return slice("2022-12-12 08:00:00", "2022-12-12 18:00:00")
        if index == "CSM4_strain":
            return slice("2022-12-09", "2022-12-15")
        return slice(None)


if __name__ == "__main__":
    seaborn().set_context("talk")

    ct = cold_test(project_dir="CSM2", read_txt=True)
    ct.load_coldtest("strain", read_txt=True)
    ct.load_coldtest("displace", read_txt=True)
    ct.load_coldtest("current", read_txt=True)

    ct.plot_row("displace", index="fit", ncol=2)
    ct.plot_loop("displace", index="fit")
    ct.fit("displace", index="test")
