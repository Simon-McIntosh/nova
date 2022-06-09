
from nova.assembly.ansysvtk import AnsysVTK


vtk = AnsysVTK(file='k1', subset='case_il')
vtk.mesh += AnsysVTK(file='k1', subset='case_ol').mesh
vtk.mesh['TFonly-cooldown'] = \
    vtk.mesh['TFonly'] - vtk.mesh['cooldown']
#vtk.mesh = vtk.mesh.clip_box([-15, 15, -15, 15, -15, 0], invert=False)
vtk.mesh = vtk.mesh.clip_box([-5, 5, -5, 5, -0.5, 0], invert=False)
#vtk.warp(factor=50)

vtk.animate('as_simulated', 'TFonly-cooldown', view='xy',
            max_factor=50, zoom=1.3, opacity=0.75)
