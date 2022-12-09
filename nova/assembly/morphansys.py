"""Generate ansys interpolation tables to apply mesh morphing to FEA."""
from dataclasses import dataclass, field
import inspect
import os

import pyvista as pv
import scipy.spatial
import scipy.interpolate

import numpy as np
from nova.definitions import root_dir
from nova.assembly.ansyspost import AnsysPost
from nova.assembly.fiducialcoil import FiducialCoil
from nova.assembly.morph import Morph
from nova.utilities import ANSYS


@dataclass
class MorphAnsys:
    """Generate Ansys morping interpolation tables."""

    name: str
    fiducial: pv.PolyData
    base: pv.PolyData
    resolution: float = 0.25
    smoothing: float = 1.0
    repo: str = 'tfc18'

    def __post_init__(self):
        """Load morphed mesh and write table."""
        self._check_root()
        if not os.path.isfile(os.path.join(self.morph_dir, self.name+'.mac')):
            self.write_table()

    @property
    def root_dir(self):
        """Return root dir."""
        return os.path.join(root_dir, f'../{self.repo}')

    @property
    def macro_dir(self):
        """Return macro dir."""
        return os.path.join(self.root_dir, 'MACRO')

    @property
    def morph_dir(self):
        """Return morph data dir."""
        return os.path.join(self.root_dir, 'MORPH')

    def _check_root(self):
        """Check for tfc18 root dir."""
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f'APDL root dir {self.root_dir} not found.'
                                    ' Pull project from gitlab')

    def cdread(self):
        """Write utility cdb initialization script."""
        txt = '''
            /clear
            cdread,db,MAGNET,cdb,../../MACRO/INPUT_FILES
            '''
        with open(os.path.join(self.macro_dir, 'read_cdb.mac'), 'w') as f:
            f.write(inspect.cleandoc(txt))

    def generate_axes(self, csys=0):
        """Return interpolation axes."""
        if csys == 0:  # cartesian
            bounds = [self.base.bounds[2*i:2*i+2] for i in range(3)]
            axes = [np.linspace(*limit, int(np.diff(limit)/self.resolution))
                    for limit in bounds]
            return axes
        index = ~np.isnan(self.base.points).any(axis=1)
        radius = np.linalg.norm(self.base.points[index, :1], axis=0)
        rlimit = [np.min(radius), np.max(radius)]
        zlimit = self.base.bounds[-2:]

        print(int(np.diff(rlimit)/self.resolution))

        axes = [np.linspace(*rlimit, int(np.diff(rlimit)/self.resolution)),
                np.linspace(0, 2*np.pi, 5000),
                np.linspace(*zlimit, int(np.diff(zlimit)/self.resolution))]
        print(np.shape(axes[0]), np.diff(rlimit), rlimit)
        return axes

    def write_table(self):
        """Write ansys interpolation table to file."""
        axes = self.generate_axes(csys=1)
        '''
        grid = np.array(np.meshgrid(*axes, indexing='ij')).T

        shape = grid.shape
        grid = grid.reshape(-1, 3)

        scipy.spatial.Delaunay()

        interpolator = scipy.interpolate.LinearNDInterpolator(
            self.fiducial.points, self.fiducial['delta'], fill_value=0.0)

        #delta = morph.rbf(grid).reshape(*shape).T

        delta = interpolator(grid)
        index = np.isnan(delta).any(axis=1)

        morph = Morph(self.fiducial, smoothing=self.smoothing)
        #delta[index] = morph.rbf(grid[index, :])

        delta = delta.reshape(*shape).T
        delta *= 500

        with ANSYS.table(os.path.join(self.morph_dir, f'{self.name}'),
                         ext='.mac') as table:
            for i, coord in enumerate('xyz'):
                table.write_text('/nopr')
                table.load(f'morph_d{coord}', delta[i], [*axes])
                table.write(['x', 'y', 'z'])
                table.write_text('/gopr')
        '''

    def write_macro(self):
        """Write ansys interpolation tables script to file."""
        txt = '''

        *get,root_type,parm,root_dir,type
        *if,root_type,eq,5,then
          /filename,'morph',0
          /cwd,'./%batchname%_%arg1%/_models'

          parsav,all
          resume,'MECH',db,,0,1
          parres
          allsel,all
        *endif

        ! load interpolation table
        *use,'../../MORPH/%arg2%%arg1%.mac'

        ! perform interpolation
        esel,all
        nsle
        *vget,nds,node,,nlist
        *vget,ndsel,node,1,nsel

        *get,nnd,node,,count
        *get,nnd_max,node,,num,maxd
        *dim,ndarray,array,nnd_max
        xyz='x','y','z'
        *do,ii,1,3
          *vget,ndarray(1),node,,loc,%xyz(ii)%
          *vmask,ndsel
          *vfun,%xyz(ii)%coord,comp,ndarray(1)
          *dim,%xyz(ii)%delta,array,nnd
        *enddo
        *do,ii,1,3
          *vitrp,%xyz(ii)%delta,morph_d%xyz(ii)%(1,1,1),xcoord(1),ycoord(1),zcoord(1)
         *enddo
        *do,ii,1,3
          *voper,%xyz(ii)%coord(1),%xyz(ii)%coord(1),add,%xyz(ii)%delta(1)
        *enddo

        ! move nodes
        /prep7
        shpp,off
        /nopr
        nmodif,nds(1:nnd),xcoord(1:nnd),ycoord(1:nnd),zcoord(1:nnd)
        /gopr
        *if,root_type,eq,5,then
          save,'./MECH', db
          finish
          /cwd,root_dir(1)
        *endif

        ! cleanup
        *del,nds
        *del,ndsel
        *del,nnd
        *del,nnd_max
        *del,ndarray
        *do,ii,1,3
          *del,%xyz(ii)%coord
          *del,morph_d%xyz(ii)%
          *del,%xyz(ii)%delta
        *enddo
        *del,xyz
        *del,ii

        '''
        with open(os.path.join(self.macro_dir, 'morph.mac'), 'w') as f:
            f.write(inspect.cleandoc(txt))


if __name__ == '__main__':

    fiducial = FiducialCoil('fiducial', 10)
    base = AnsysPost('TFCgapsG10', 'k0', 'all')

    ansys = MorphAnsys('ccl0', fiducial.mesh, base.mesh)

    #ansys.write_table()
    #ansys.write_macro()
