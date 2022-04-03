import itertools
import pickle
import datetime
import os

import numpy as np
import matplotlib.image as mpimg

from nova.utilities.pyplot import plt


def log_coef(xlim):
    b = np.log10(xlim[1] / xlim[0]) / (xlim[1] - xlim[0])
    a = xlim[1] / 10**(b * xlim[1])
    return a, b


def log2lin(x, xlim):
    a, b = log_coef(xlim)
    x = np.array(x)
    x = a * 10**(b * x)
    return x


def lin2log(x, xlim):
    a, b = log_coef(xlim)
    x = np.log10(x / a) / b
    return x


def stripfile(file):
    file = file.replace('.png', '')  # strip file type
    file = file.replace('.jpg', '')
    return file


class sample_plot(object):

    def __init__(self, data, x_origin, y_origin, x_ref, y_ref,
                 x_fig, y_fig, ax_eq, ax, fig, path, file,
                 xscale='linear', yscale='linear', save=True):
        self.points = []
        self.points_data = []
        self.xscale = xscale
        self.yscale = yscale
        self.cord_flag = 0
        self.count_data = itertools.count(1)
        self.data = data
        self.path = path
        self.file = file
        self.save = save
        self.set_folder()
        self.x = []
        self.y = []
        self.xy = []
        self.x_fig = x_fig
        self.y_fig = y_fig
        self.ax_eq = ax_eq
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.ax = ax
        self.fig = fig
        self.cycle_click = itertools.cycle([1, 2])
        self.cycle_axis = itertools.cycle([1, 2, 3, 4])
        self.count_click = next(self.cycle_click)
        self.count_axis = next(self.cycle_axis)
        self.exit = False
        self.cid = x_origin.figure.canvas.mpl_connect(
            'button_press_event', self)
        if self.ax_eq == -1:
            self.count_axis = next(self.cycle_axis)
            self.count_axis = next(self.cycle_axis)

    def set_folder(self):
        self.file = stripfile(self.file)
        date_str = datetime.date.today().strftime('%Y_%m_%d')  # today
        self.folder = self.path + 'imagedata/'
        if not os.path.exists(self.folder) and self.save:
            os.makedirs(self.folder)
        self.file = date_str + '_' + self.file + '.pkl'

    def __call__(self, event):
        if event.button == 1:  # select button
            if self.count_click == 1:  # enter axis
                if self.count_axis == 1:
                    self.x_origin.set_data(event.xdata, event.ydata)
                    set_title(self.ax, self.x_fig[1], 'x-referance')
                    self.x_origin.figure.canvas.draw()
                elif self.count_axis == 2:
                    self.x_ref.set_data(event.xdata, event.ydata)
                    if self.ax_eq == 1:
                        self.ax.set_title('select data (select|finish|undo)')
                        self.count_click = next(self.cycle_click)
                    else:
                        set_title(self.ax, self.y_fig[0], 'y-origin')
                    self.x_ref.figure.canvas.draw()
                elif self.count_axis == 3:
                    self.y_origin.set_data(event.xdata, event.ydata)
                    set_title(self.ax, self.y_fig[1], 'y-referance')
                    self.y_origin.figure.canvas.draw()
                else:
                    self.y_ref.set_data(event.xdata, event.ydata)
                    self.ax.set_title('select data (select|finish|undo)')
                    self.y_ref.figure.canvas.draw()
                    self.count_click = next(self.cycle_click)
                self.count_axis = next(self.cycle_axis)
            else:  # enter data
                self.x.append(event.xdata)
                self.y.append(event.ydata)
                self.data.set_data(self.x, self.y)
                self.data.figure.canvas.draw()
        if event.button == 2:  # enter button
            if len(self.x) == 0:  # exit and save (dual click)
                self.fig.canvas.mpl_disconnect(self.cid)
                plt.close(self.fig)
                self.exit = True
            else:
                if self.cord_flag == 0:
                    self.set_cord()
                    self.cord_flag = 1
                self.store_data()
                self.x, self.y = [], []  # reset
                self.data.set_data(self.x, self.y)
                self.data.figure.canvas.draw()
        if event.button == 3:  # remove data points
            if self.count_click == 2:
                if len(self.x) > 0:
                    self.x.pop(len(self.x) - 1)
                if len(self.y) > 0:
                    self.y.pop(len(self.y) - 1)
                self.data.set_data(self.x, self.y)
                self.data.figure.canvas.draw()

    def set_cord(self):
        if self.ax_eq == 1:  # referance from x-dir
            x_ref = self.x_ref.get_xydata()[0][0]
            self.x_o = self.x_origin.get_xydata()[0]
            self.x_o[1] = -self.x_o[1]
            self.y_o = self.x_o
            self.x_scale = (
                self.x_fig[1] - self.x_fig[0]) / (x_ref - self.x_o[0])
            self.y_scale = self.x_scale
        elif self.ax_eq == -1:  # referance from y-dir
            y_ref = -self.y_ref.get_xydata()[0][1]
            self.y_o = self.y_origin.get_xydata()[0]
            self.y_o[1] = -self.y_o[1]
            self.x_o = self.y_o
            self.y_scale = (
                self.y_fig[1] - self.y_fig[0]) / (y_ref - self.y_o[1])
            self.x_scale = self.y_scale
        else:  # referance from x and y
            x_ref = self.x_ref.get_xydata()[0][0]
            y_ref = -self.y_ref.get_xydata()[0][1]
            self.x_o = self.x_origin.get_xydata()[0]
            self.x_o[1] = -self.x_o[1]
            self.y_o = self.y_origin.get_xydata()[0]
            self.y_o[1] = -self.y_o[1]
            self.x_scale = (
                self.x_fig[1] - self.x_fig[0]) / (x_ref - self.x_o[0])
            self.y_scale = (
                self.y_fig[1] - self.y_fig[0]) / (y_ref - self.y_o[1])

    def store_data(self):
        points_data = self.data.get_xydata()
        x = points_data[:, 0]
        y = -points_data[:, 1]
        x = self.x_scale * (x - self.x_o[0]) + self.x_fig[0]
        y = self.y_scale * (y - self.y_o[1]) + self.y_fig[0]
        if self.xscale == 'log':
            x = log2lin(x, self.x_fig)
        if self.yscale == 'log':
            y = log2lin(y, self.y_fig)
        self.points.append({'x': x, 'y': y})
        self.points_data.append(points_data)
        limits = {}
        for var in ['x_o', 'x_scale', 'x_fig', 'y_o', 'y_scale', 'y_fig',
                    'xscale', 'yscale']:
            limits[var] = getattr(self, var)
        if self.save:
            with open(self.folder + self.file, 'wb') as output:
                pickle.dump(self.points, output, -1)
                pickle.dump(self.points_data, output, -1)
                pickle.dump(limits, output, -1)


def set_format(var):
    if abs(var) > 1e4:
        f = '1.3e'
    else:
        f = '1.3f'
    return f


def set_title(ax, var, txt):
    f = set_format(var)
    axis = txt[0]
    mode = txt[2:]
    ax.set_title(f'select {axis}-{mode} ({axis}={var:{f}})')


def data_mine(path, file, xlim, ylim, **kw):
    label = kw.get('label', '')
    save = kw.get('save', True)
    title = kw.pop('title', None)
    if 'scale' in kw:
        scale = kw.get('scale')
        xscale = scale
        yscale = scale
    else:
        xscale = kw.get('xscale', 'linear')
        yscale = kw.get('yscale', 'linear')
    x_fig, y_fig = xlim, ylim
    ax_eq = 0

    if len(x_fig) == 0:
        ax_eq = -1
        x_fig = y_fig

    if len(y_fig) == 0:
        ax_eq = 1
        y_fig = x_fig

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)
    plt.ion()

    origin = 'upper'
    if '.png' not in file and '.jpg' not in file:
        file += '.png'
    image = mpimg.imread(os.path.join(path, file))
    ax.imshow(image, origin=origin)

    if ax_eq == -1:
        set_title(ax, y_fig[0], 'y-origin')
    else:
        set_title(ax, x_fig[0], 'x-origin')

    # default markers
    data, = ax.plot([0], [0], 'C1d-')
    x_origin, = ax.plot([0], [0], 'C3o')
    y_origin, = ax.plot([0], [0], 'C4o')
    x_ref, = ax.plot([0], [0], 'C3s')
    y_ref, = ax.plot([0], [0], 'C4s')

    if label:
        file += f'_{label}'
    if title:
        plt.suptitle(title)

    plot = sample_plot(data, x_origin, y_origin, x_ref, y_ref,
                       x_fig, y_fig, ax_eq, ax, fig, path, file,
                       xscale=xscale, yscale=yscale, save=save)
    while not plot.exit:
        plt.pause(1)
    return plot


def get_filename(path, file, **kwargs):
    date = kwargs.get('date', datetime.date.today().strftime('%Y_%m_%d'))
    label = kwargs.get('label', '')
    file = stripfile(file)
    filename = path + 'imagedata/' + date + '_' + file
    if label:
        filename += f'_{label}'
    return filename


def data_load(path, filename, **kwargs):
    filepath = get_filename(path, file=filename, **kwargs)
    with open(filepath + '.pkl', 'rb') as input:
        points = pickle.load(input)
        points_data = pickle.load(input)
        limits = pickle.load(input)
    return points, points_data, limits


def image_plot(path, file, ax=None, **kwargs):
    limits = data_load(path, file, **kwargs)[2]
    for axis in ['x', 'y']:
        lim = limits[f'{axis}_fig']
        '''
        if limits[f'{axis}scale'] == 'log':
            delta = limits[f'{axis}_o'][1] - limits[f'{axis}_o'][0]
            limits[f'{axis}_scale'] *= delta
            limits[f'{axis}_scale'] = log2lin(limits[f'{axis}_scale'], lim)
            limits[f'{axis}_scale'] /= delta
            limits[f'{axis}_fig'] = log2lin(limits[f'{axis}_fig'], lim)
        '''
        if limits[f'{axis}scale'] == 'log':
            scale = limits[f'{axis}_fig'][1]-limits[f'{axis}_fig'][0]
            limits[f'{axis}_fig'] = np.log10(limits[f'{axis}_fig'])
            scale /= (limits[f'{axis}_fig'][1]-limits[f'{axis}_fig'][0])
            limits[f'{axis}_scale'] /= scale

    if ax is None:
        ax = plt.subplots(1, 1, figsize=(8, 10))[1]
    image = mpimg.imread(path + file + '.png')
    extent = np.zeros(4)  # scale image extent
    extent[0] = -limits['x_scale'] * limits['x_o'][0] + limits['x_fig'][0]
    extent[1] = limits['x_scale'] * (np.shape(image)[1] - limits['x_o'][0]) +\
        limits['x_fig'][0]
    extent[3] = -limits['y_scale'] * limits['y_o'][1] + limits['y_fig'][0]
    extent[2] = -limits['y_scale'] * (np.shape(image)[0] + limits['y_o'][1]) +\
        limits['y_fig'][0]
    ax.imshow(image, origin='upper', extent=extent)
    aspect = (np.shape(image)[1] / np.shape(image)[0])
    force_aspect(ax, aspect)
    return ax


def force_aspect(ax, aspect=1):
    # aspect is width/height
    scale_str = ax.get_yaxis().get_scale()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if scale_str == 'linear':
        asp = abs((xmax-xmin) / (ymax-ymin)) / aspect
    elif scale_str == 'log':
        asp = abs((np.log(xmax) - np.log(xmin)) /
                  (np.log(ymax) - np.log(ymin))) / aspect
    ax.set_aspect(asp)
