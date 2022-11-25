
import os

#from ansys.dpf import core as dpf

#from ansys.dpf import post
import ansys.dpf.core as dpf

from nova.definitions import root_dir
from nova.plot import plt


from ansys.mapdl.core import launch_mapdl


mapdl = launch_mapdl()


'''

rst_file = os.path.join(root_dir, 'data/Ansys/TFC18/V4.rst')

model = dpf.Model(rst_file)

results = model.results
displacements = results.displacement()


select = model.operator('scoping_provider_by_ns')
select.inputs.requested_location.connect('elem')
select.inputs.named_selection_name.connect('WP')
#select.inputs.data_sources.connect(model.metadata.data_sources)


#results = model.results
#displacements = results.displacement()
#fields = displacements.outputs.fields_container()

#vol = model.results.volume()
#field = vol.outputs.fields_container()[0]

#oper = dpf.Operator('U')
#oper.inputs.data_sources(data)

'''

'''
#solution = post.load_solution()
#mesh = solution.mesh

#post.print_available_keywords()
displacement = solution.displacement(
    location=post.locations.nodal, time_scoping=[1],
    element_scoping=
    grouping=post.grouping.by_material)
displacement.norm.plot_contour(show_edges=False, nan_opacity=0)

#post.grouping.by_body = 'WP'
#displacement = solution.stress(named_selection='WP')

#location=post.locations.nodal, time_scoping=[1],


#ids = x.get_scoping_at_field()
#solution.mesh.nodes.map_scoping(x.result_fields_container[0].scoping)
#coordinates = solution.mesh.nodes.coordinates_field.data[ids]

#plt.plot(coordinates[:, 0], coordinates[:, 2], '.')

#
#displacement.tensor.plot_contour(show_edges=False, nan_opacity=0)
'''
