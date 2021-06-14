
from ansys.mapdl.core import Mapdl
from ansys.mapdl.core import launch_mapdl

mapdl = Mapdl()
# mapdl = launch_mapdl(additional_switches='-smp')

mapdl.post1()
#mapdl.file('V4', 'rst')
#mapdl.set(3, 2)
mapdl.esel('s', 'elem', '', 'WP')
mapdl.nsle()
#mapdl.eplot(show_edges=False)
#mapdl.plnsol('u', 'sum')
#mapdl.pldisp()

#result = mapdl.result

#mapdl.set(3, 2)
mapdl.nsel('all')
mapdl.slashdscale(200)
mapdl.post_processing.plot_nodal_displacement()

#result.plot_nodal_displacement(0, show_edges=False)
