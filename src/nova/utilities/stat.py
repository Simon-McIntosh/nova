import numpy as np
from amigo.pyplot import plt


class histopeaks:  # extract peaks from discrete PDF data

    def __init__(self, time, x, nbins=51, nlim=3, nstd=None):
        self.time = time
        self.x = x
        self.get_hist(nstd, nbins)
        self.get_peaks(nlim=nlim)

    def inband(self, x, b, expand):
        c, w = self.peaks['centre'][b], self.peaks['width'][b]
        return x >= c-expand*w/2 and x <= c+expand*w/2

    def getband(self, x, expand):
        b = np.argmin(abs(self.peaks['centre']-x))
        if not self.inband(x, b, expand):
            b = -1
        return b

    def timeseries(self, expand=1.5, plot=False, **kwargs):
        self.expand = expand
        self.Ip = kwargs.get('Ip', self.x)  # plasma current
        b = -1  # start outside band
        index = [[] for _ in range(self.npeak)]  # list of lists
        for i, (t, x) in enumerate(zip(self.time, self.x)):
            bo = b  # inital state
            if b >= 0:  # inband
                if not self.inband(x, b, self.expand):  # band-band, exit
                    index[b][-1][1] = i  # record exit
                    b = self.getband(x, self.expand)  # update band
            else:
                b = self.getband(x, self.expand)  # update band
            if b != bo and b != -1:  # record entry
                index[b].append([i, 0])
        index[b][-1][1] = i  # exit
        self.mode_imax = np.zeros((self.npeak, 2), dtype=int)  # sort
        for i, b in enumerate(range(self.npeak)):
            subindex = np.array(index[b])
            isort = np.argsort(np.diff(subindex).flatten())
            index[b] = subindex[isort][::-1]
            try:
                self.mode_imax[i] = index[b][0]
            except IndexError:  # empty index
                self.mode_imax[i] = 0
        self.mode_index = index
        # self.mode_imax = self.mode_imax[:-1]
        if plot:
            self.plot_timeseries()
        return self.mode_imax

    def plot_timeseries(self, **kwargs):
        ax = plt.subplots(2, 1, sharex=True)[1]
        ax[0].plot(self.time, 1e-6*self.x)
        for ind in self.mode_imax:
            ax[0].plot(self.time[ind[0]:ind[1]], 1e-6*self.x[ind[0]:ind[1]])
        nmax, nmin = 0, 0
        for peak in self.peaks:
            center, width = peak['centre'], peak['width']
            ax[0].plot(self.time, 1e-6*center*np.ones(self.nsum),
                       '--', color='gray')
            ax[0].fill_between(
                self.time,
                1e-6*(center+self.expand*width/2)*np.ones(self.nsum),
                1e-6*(center-self.expand*width/2)*np.ones(self.nsum),
                color='gray', alpha=0.2)
            if nmin > center-self.expand*width/2:
                nmin = center-self.expand*width/2
            if nmax < center+self.expand*width/2:
                nmax = center+self.expand*width/2
        dlim = nmax-nmin
        ax[0].set_ylim(1e-6*np.array([nmin-0.1*dlim, nmax+0.1*nmax]))
        ax[0].set_ylabel(r'$dI/dt$ MAs$^{-1}$')
        Ip = kwargs.get('Ip', self.Ip)  # plasma current
        ax[1].plot(self.time, 1e-6*Ip, color='lightgray', zorder=-100)
        for i, ind in enumerate(self.mode_imax):
            color = 'C{:d}'.format((i+1) % 10)
            ax[1].plot(self.time[ind[0]:ind[1]], 1e-6*Ip[ind[0]:ind[1]],
                       color=color, zorder=-i)
            if i == 0:
                ax[1].plot([self.time[ind[0]], self.time[ind[1]]],
                           1e-6*np.array([Ip[ind[0]], Ip[ind[1]]]),
                           'o', color='C{:d}'.format(i+1))
                dt = self.time[ind[1]]-self.time[ind[0]]
                ax[1].text(self.time[ind[1]], 1e-6*Ip[ind[1]],
                           ' flattop $\Delta t$={:1.0f}s'.format(dt),
                           ha='left', va='bottom',
                           color='C{:d}'.format(i+1))
        ax[1].set_xlabel(r'$t$ s')
        ax[1].set_ylabel(r'$I_p$ MA')
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)

    def get_hist(self, nstd, nbins):
        if nstd is None:  # consider full range
            xrange = np.array([np.min(self.x), np.max(self.x)])
        else:
            xstd = np.std(self.x)
            xrange = nstd*xstd*np.array([-1, 1]) + np.mean(self.x)
        self.nsum = len(self.x)
        self.nhist, self.bins = np.histogram(self.x, bins=nbins, range=xrange)
        nmin = int(0.001*np.max(self.nhist))
        self.nhist[self.nhist < nmin] = nmin

    def plot_hist(self):
        plt.figure()
        plt.hist(self.x, self.bins)

    def count_peaks(self, nx):
        i_rise, i_fall = 0, 0
        valley = True
        peaks = {'centre': [], 'width': [], 'nmax': [], 'npeak': []}
        n = 0
        while True:
            if valley:
                i_rise = next((i+i_fall for i, n in
                               enumerate(self.nhist[i_fall:]) if n > nx), None)
                valley = False
            else:
                i_fall = next((i+i_rise for i, n in
                               enumerate(self.nhist[i_rise:]) if n < nx), None)
                valley = True
                if i_fall is not None:
                    n += 1
                    edge = np.array([self.bins[i_rise], self.bins[i_fall]])
                    mean = np.sum(self.bins[i_rise:i_fall] *
                                  self.nhist[i_rise:i_fall]) /\
                        np.sum(self.nhist[i_rise:i_fall])
                    peaks['centre'].append(mean)
                    peaks['width'].append(np.diff(edge)[0])
                    peaks['nmax'].append(np.max(self.nhist[i_rise:i_fall]))
                    peaks['npeak'].append(np.sum(self.nhist[i_rise:i_fall]))
            if i_rise is None or i_fall is None:
                break
        peaks['n'] = n
        return peaks

    def get_peaks(self, nlim=None, plot=False):
        nmax = 0
        for nx in np.sort(self.nhist):
            peaks = self.count_peaks(nx)
            if peaks['n'] > nmax:
                nmax = peaks['n']
                peaks['nx'] = nx
                peaks_o = peaks  # store
        if nmax == 0:
            raise ValueError('no peaks found - check input data')
        self.peaks = np.zeros(nmax, dtype=[('centre', float),
                                           ('width', float),
                                           ('nmax', int),
                                           ('npeak', int)])
        if nlim is not None and nmax > nlim:  # apply band limit
            nmax = nlim
        for key in self.peaks.dtype.names:
            self.peaks[key] = peaks_o[key]
        self.peaks.sort(order='nmax')
        self.peaks = self.peaks[::-1][:nmax]
        self.nx = peaks_o['nx']
        self.npeak = nmax
        if plot:
            self.plot_peaks()
        return self.peaks

    def plot_peaks(self):
        plt.figure()
        dbin = np.mean(np.diff(self.bins))
        xbin = self.bins[:-1]+dbin/2
        plt.bar(xbin, self.nhist, width=0.95*dbin)

        plt.plot(self.bins, self.nx*np.ones(len(self.bins)),
                 '--', color='gray')
        plt.text(self.bins[-1], self.nx, 'Nclip {:1.0f}'.format(self.nx),
                 ha='left', va='bottom')
        for i, (centre, width, nmax, npeak) in \
                enumerate(zip(self.peaks['centre'], self.peaks['width'],
                              self.peaks['nmax'], self.peaks['npeak'])):
            plt.plot(centre*np.ones(2), np.array([0, nmax]), '-.C1')
            plt.text(centre, nmax,
                     'P{:d} {:1.1f}%'.format(i, 100*npeak/self.nsum),
                     ha='center', va='bottom')
            for x in width/2*np.array([-1, 1]):
                plt.plot((centre+x)*np.ones(2), np.array([0, nmax]),
                         '--', color='C1')
        plt.xlabel('$x$')
        plt.ylabel('$N(x)$')
        plt.despine()
