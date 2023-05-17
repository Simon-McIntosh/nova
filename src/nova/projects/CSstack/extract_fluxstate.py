import os
import nova
from amigo.png_tools import data_mine, data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
from amigo.IO import pythonIO
import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Polygon
from descartes import PolygonPatch
from labellines import labelLine


class fluxmap(pythonIO):
    
    def __init__(self, filename='Hmode_pedestal', xvar='fluxstate'):
        self.path = os.path.join(class_dir(nova), '../Inputs/fluxmap/')
        self.set_xvar(xvar)
        self.load(filename)
        
    def set_xvar(self, xvar):
        if xvar not in ['fluxstate', 'Li3']:
            raise IndexError(f'xvar {xvar} not in [fluxstate, Li3]')
        self.xvar = xvar
        
    def get_xy(self):
        xy = ['Psi', 'Li3']
        if self.xvar == 'Li3':
            xy = xy[::-1] 
        return xy 
        
    def extract(self, filename, xlim, ylim):
        'extract datapoints from image'
        data_mine(self.path, filename, xlim, ylim)
        
    def package_Hmode_pedestal(self):
        points = self.load_points('ITER_PF_15MA_with_pedestal', 
                                  date='2020_06_17')
        self.limits = {}
        self.limits['access'] = {
                'Li3': points[6]['x'], 'Psi': points[6]['y'],
                'limit': r'access from IM, $C_{Ejima}$=0.5'}
        self.limits['low Li SOB'] = {
                'Li3': points[0]['x'], 'Psi': points[0]['y'],
                'limit': r'dome gap, $I_{PF6}$ & $F_{sep 3L/2L}$'}
        self.limits['low Li EOB'] = {
                'Li3': np.append(points[2]['x'][:-2], points[4]['x'][0]),
                'Psi': np.append(points[2]['y'][:-2], points[4]['y'][0]),
                'limit': r'upper shape change, $I_{CS1/2}$ & $F_{z}$'}
        self.limits['high Li EOB relax'] = {
                'Li3': points[4]['x'], 'Psi': points[4]['y'],
                'limit': r'relaxed shape, $I_{CS1/2}$ & $F_{z}$'}
        self.limits['high Li EOB'] = {
                'Li3': np.append(points[4]['x'][0], points[5]['x'][1:]),
                'Psi': np.append(points[4]['y'][0], points[5]['y'][1:]),
                'limit': r'good shape, $I_{CS1/2}$ & $F_{z}$'}
        Psi = interp1d(self.limits['access']['Li3'][:-1], 
                       self.limits['access']['Psi'][:-1],
                       fill_value='extrapolate')
        self.limits['access']['Li3'] = np.append(
                self.limits['access']['Li3'], 0.65)
        self.limits['access']['Psi'] = np.append(
                self.limits['access']['Psi'], Psi(0.65))
        # solve access - dome intersection
        access_points = [(Li3, Psi) 
                         for Li3, Psi in zip(
                                 self.limits['access']['Li3'], 
                                 self.limits['access']['Psi'])]
        access_line = LineString(access_points)
        dome_points = [(Li3, Psi) 
                       for Li3, Psi in zip(
                               self.limits['low Li SOB']['Li3'], 
                               self.limits['low Li SOB']['Psi'])]
        dome_line = LineString(dome_points)
        lower_corner = dome_line.intersection(access_line).xy
        lc = {'Li3': lower_corner[0][0], 'Psi': lower_corner[1][0]}
        # trim access
        self.limits['access']['Li3'][-1] = lc['Li3']
        self.limits['access']['Psi'][-1] = lc['Psi']
        # trim dome
        #self.limits['low Li SOB']['Li3'][0] = lc['Li3']
        #self.limits['low Li SOB']['Psi'][0] = lc['Psi']
        self.limits['low Psi'] = {
                'Li3': [1, self.limits['low Li SOB']['Li3'][0]],
                'Psi': [self.limits['low Li SOB']['Psi'][0]-5, 
                        self.limits['low Li SOB']['Psi'][0]-5],
                'limit': None}
        self.save('Hmode_pedestal')
        
    def save(self, filename):
        attributes = ['limits']
        filename = os.path.join(self.path, filename)
        self.save_pickle(filename, attributes)
        
    def load(self, filename):
        if filename is not None:
            filename = os.path.join(self.path, filename)
            self.load_pickle(filename)
        
    def load_points(self, filename, date=None):
        if date is None:
            kwargs = {}
        else:
            kwargs = {'date': date}
        points = data_load(self.path, filename, **kwargs)[0]
        return points
    
    def get_patch(self, relax=False):
        points = {'Li3': [], 'Psi': []}
        for p in points:
            if relax:
                points[p].extend(self.limits['low Li SOB'][p])
                points[p].extend(self.limits['low Li EOB'][p])
                points[p].extend(self.limits['high Li EOB relax'][p])
            else:
                points[p].extend(self.limits['low Li SOB'][p][:2])
                points[p].extend(self.limits['low Li EOB'][p][-2:])
                points[p].extend(self.limits['high Li EOB'][p])
            # points[p].extend(self.limits['access'][p])
            points[p].extend(self.limits['low Psi'][p])
        xvar, yvar = self.get_xy()
        loop = [(x, y) for x, y in zip(points[xvar], points[yvar])]
        patch = PolygonPatch(Polygon(loop))
        patch.set_linewidth(0)
        if relax:
            patch.set_facecolor('darkgray')
        else:
            patch.set_facecolor('lightgray')
        return patch
    
    def plot_patch(self, relax=False, ax=None):
        if ax is None:
            ax = plt.gca()
        patch = self.get_patch(relax=relax)
        ax.add_patch(patch)
        
    def plot_limits(self, ax=None):
        if ax is None:
            ax = plt.gca()
        xvar, yvar = self.get_xy()
        iC = 0
        for limit in self.limits:
            if self.limits[limit]['limit'] is not None:
                if limit == 'access':
                    ls, lw, alpha = 'k--', 2, 0.5
                    label = None
                else:
                    ls, lw, alpha = f'C{iC}:', 4, 1
                    iC += 1
                    label = self.limits[limit]['limit']
                ax.plot(self.limits[limit][xvar], 
                        self.limits[limit][yvar], ls,
                        label=label, lw=lw, alpha=alpha)
                if limit == 'access':
                    labelLine(ax.get_lines()[-1], 122, 
                              label='$C_{Ejima}$=0.5',
                              ha='center', va='center', align=True,
                              fontsize='xx-small',
                              bbox={'facecolor': 'lightgray', 'ec': None,
                                    'lw': 0})
            
    def set_labels(self, ax=None):
        if ax is None:
            ax = plt.gca()
        labels = ['$Li(3)$', '$<\Psi_{coils}>$, Wb']
        if self.xvar == 'fluxstate':
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[0])
        else:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
        self.legend = ax.legend(loc='upper center', 
                                 bbox_to_anchor=(0.5, 1.2), 
                                 ncol=2, frameon=True, fontsize='x-small')
        plt.despine()
        
    def plot(self, ax=None):
        self.plot_patch(relax=True, ax=ax)
        self.plot_patch(relax=False, ax=ax)
        self.plot_limits(ax=ax)
        self.set_labels(ax=ax)
        
if __name__ == '__main__':
    
    fmap = fluxmap()
    fmap.package_Hmode_pedestal()
    plt.set_context('talk')
    fmap.plot()


   
    

"""

from nep.DINA.read_scenario import scenario_data, field_data
from amigo.geom import vector_lowpass as lp
from amigo.pyplot import plt 

# overplot scenario data
d2 = scenario_data(read_txt=False)
dt_filt = 15

iC = 0
folders = [f for f in d2.folders if '15MA DT-DINA' in f]
#sns.set_palette('Set2')
folders = ['15MA DT-DINA2014-01'] + folders[-11:]
files = []
for file in folders:
    d2.load_file(file)  # read / load single file
    n_filt = int(dt_filt/d2.dt) 
    if n_filt%2 == 0:
        n_filt += 1
    try:
        S = int(d2.feature_keypoints.loc['SOB', 'frame_index'])
        E = int(d2.feature_keypoints.loc['EOB', 'frame_index'])
        Li_3 = lp(d2.frame.loc[S:E, 'li(3)'].values, n_filt)
        Psi = lp(d2.frame.loc[S:E, '<PSIcoils>'].values, n_filt)
        #Psi = lp(d2.frame.loc[S:E, 'PSI(axis)'].values, n_filt)
        #Psi = lp(d2.frame.loc[S:E, '<PSIext>'].values, n_filt)
        dt = d2.frame.loc[E, 't'] - d2.frame.loc[S, 't'] 
        if dt > 50:
            iC = iC%9
            if Psi[0] < 0:
                Psi *= -1
            
            ax.plot(Li_3[0], Psi[0], f'C{iC}o')
            ax.plot(Li_3, Psi, f'C{iC}', 
                     label=file.replace('15MA DT-DINA', ''))
            ax.plot(Li_3[-1], Psi[-1], f'C{iC}s')
            iC += 1
            files.append(file)
    except KeyError:
        pass
    

plt.despine()
plt.xlabel('$Li(3)$')
plt.ylabel('$<\Psi_{coils}>$ Wb')
h = ax.get_legend_handles_labels()[0]
plt.legend(handles=h[5:], ncol=1, loc='center right', 
           bbox_to_anchor=(1.4, 0.5), frameon=False)

ax.add_artist(limit_legend)

d3 = field_data(read_txt=False)

ax = plt.subplots(1, 1)[1]
for file in files:
    d3.load_file(file)
    ax.plot(d3.frame.loc[:, 'Fz_2'], label=file)
plt.legend(ncol=1, loc='center right', 
           bbox_to_anchor=(1.4, 0.5), frameon=False)

ax = plt.subplots(1, 1)[1]
for file in files:
    d2.load_file(file) 
    S = int(d2.feature_keypoints.loc['XPF', 'frame_index'])
    E = int(d2.feature_keypoints.loc['SOB', 'frame_index'])
    ax.plot(d2.frame.loc[S:E, 'li(3)'],
               d2.frame.loc[S:E, 'Cejima'])
plt.despine()
plt.xlabel('$Li(3)$')
plt.ylabel('$C_{Ejima}$')
"""
