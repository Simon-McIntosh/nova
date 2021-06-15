
import os

#from ansys.mapdl.core import Mapdl
import ansys.dpf.core as dpf
from ansys.dpf import post

from nova.definitions import root_dir

rst_file = os.path.join(root_dir, 'data\\Ansys\\V4\\MAGNET_MECH')

'''
mapdl = Mapdl()
# mapdl = launch_mapdl(additional_switches='-smp')

mapdl.post1()
#mapdl.file(rst_file, 'rst')
#mapdl.set(3, 2)
mapdl.esel('s', 'elem', '', 'WP')
mapdl.nsle()
#mapdl.eplot(show_edges=False)
#mapdl.plnsol('u', 'sum')
#mapdl.pldisp()

#result = mapdl.result

#mapdl.set(3, 2)
mapdl.nsel('all')
#mapdl.slashdscale(200)
#mapdl.post_processing.plot_nodal_displacement()

#result.plot_nodal_displacement(0, show_edges=False)

#model = dpf.Model(f'{rst_file}.rst')
#results = model.results
#displacements = results.displacement()
'''

model = dpf.Model(f'{rst_file}.rst')
mesh_scoping = model.operator('scoping_provider_by_ns')
mesh_scoping.inputs.requested_location.connect('Nodal')
mesh_scoping.inputs.named_selection_name.connect('WP')
mesh_scoping.inputs.data_sources.connect(model.metadata.data_sources)
wp_scoping = mesh_scoping.outputs.mesh_scoping()

op = dpf.Operator('mesh::by_scoping')  # operator instantiation
op.inputs.scoping.connect(wp_scoping)
op.inputs.mesh.connect(model.metadata.meshed_region)
wp_mesh = op.outputs.mesh()

op = dpf.Operator('meshed_skin_sector_triangle')
op.inputs.connect(wp_mesh)
wp_skin = op.outputs.mesh()
node_scoping = op.outputs.nodes_mesh_scoping()

solution = post.load_solution(f'{rst_file}.rst')
disp = solution.displacement(set=2, node_scoping=node_scoping)
disp.norm.plot_contour(show_edges=False, opacity=1, nan_opacity=0,
                       use_transparency=False, style='surface')
