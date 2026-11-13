"""Inspect the scenario's static geometry and dynamic array inventories."""

import json
import sys

import imas
import numpy as np


SOURCE = '/work/imas/shared/imasdb/ITER/3/105027/1'
DD = '3.38.1'


def main():
    with imas.DBEntry(f'imas:hdf5?path={SOURCE}', 'r', dd_version=DD) as entry:
        names = ('pf_active', 'wall', 'equilibrium', 'pf_passive')
        occurrences = {name: entry.list_all_occurrences(name) for name in names}
        print('OCCURRENCES', occurrences, flush=True)
        ids = {name: entry.get(name, 0, lazy=False, autoconvert=False)
               for name in names if 0 in occurrences[name]}
    active = ids['pf_active']
    wall = ids['wall']
    eq = ids['equilibrium']
    print('VERSIONS', {name: str(value.ids_properties.version_put.data_dictionary)
                       for name, value in ids.items()}, flush=True)
    print('TIME', len(eq.time), np.asarray(eq.time)[[0, -1]],
          'homogeneous', int(eq.ids_properties.homogeneous_time), flush=True)
    for i in np.linspace(0, len(eq.time_slice)-1, 10).astype(int):
        s = eq.time_slice[i]
        print('SLICE', i, float(eq.time[i]), float(s.global_quantities.ip),
              'profiles2d', len(s.profiles_2d), flush=True)
    for i, coil in enumerate(active.coil):
        print('COIL', i, str(coil.name), len(coil.current.data), flush=True)
        for element in coil.element:
            g = element.geometry
            print('ELEMENT', float(element.turns_with_sign), int(g.geometry_type),
                  'outline', len(g.outline.r), 'rectangle',
                  [float(getattr(g.rectangle, k)) for k in ('r', 'z', 'width', 'height')], flush=True)
    for d in wall.description_2d:
        for u in d.limiter.unit:
            print('WALL', str(u.name), len(u.outline.r), flush=True)
    for i in np.linspace(0, len(eq.time_slice)-1, 3).astype(int):
        s = eq.time_slice[i]
        for p in s.profiles_2d:
            print('GRID', i, len(p.grid.dim1), len(p.grid.dim2),
                  'j_tor', np.asarray(p.j_tor).shape, 'psi', np.asarray(p.psi).shape, flush=True)
    print(json.dumps({'source': SOURCE, 'dd': DD}), flush=True)


if __name__ == '__main__':
    main()
