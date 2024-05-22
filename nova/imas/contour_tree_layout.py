"""Develop layout for contour_tree within equilibrium ids."""

from pprint import pprint

tree: dict[str, dict] = {
    "contour_tree": "A structure to store the location, "
    "value, and connectivity of poloidal flux critical points."
}
tree["node [structure]"] = {
    "cricical_type [:]": "0-minimum, 1-saddle, 2-maximum",
    "radius [:]": "float",
    "height [:]": "float",
    "psi [:]": "float",
}
tree["edge [structure"] = "[int, int]"
tree["x_point_index [:]"] = "list[int]"
tree["o_point_index [:]"] = "list[int]"


tree: dict[str, dict] = {
    "doc": "A structure to describe the topological connectivity of the "
    "poloidal flux map critical points as an undirected graph. "
}
tree["node [structure]"] = {
    "doc": "Nodes of a contour tree are ether local extremum (o-points) "
    "or saddle points (x-points) of the poloidal flux map. "
    "A contour tree node is defined by its critical type and point index. "
    "A critical type of 1 references a saddle topology with the point index "
    "identifying node in the x-point array of structures. "
    "A critical type of 0 or 2 references a local extremum with the point index "
    "identifying node in the o-point array of structures.",
    "critical_type [:]": "0-minimum, 1-saddle, 2-maximum",
    "point_index [:]": "int",
}
tree["edge [structure"] = {
    "doc": "Edges connect nodes to one another. "
    "A single node may connect to multiple edges such as the case "
    "where a single maximum (a hill) may topologicaly link to multiple minima "
    "(the floors of different valleys).",
    "node_index [:]": "[int, int]",
}

time_slice = {}
time_slice["x_point [array of structures]"] = {
    "doc": "Saddle points in the poloidal flux map. "
    "These points have zero gradients in orthogonal directions but are not "
    "local extremum of poloidal flux map. "
    "These points are rarely coincident with nodes defining the poloidal upon "
    "which the poloidal flux map is defined."
    "X-points are ordered first by proximitly to the magnetic axis "
    ",mesured by the number of edge conections required to link the point in question "
    "to the magnetic axis, and then by "
    "the value of poloidal flux multiplied by the sign of plasma current. A plasma "
    "current of zero is taken to be positive. For cases without a magnetic axis points "
    "are ordered soley on the value of poloidal flux. If present, the primary x_point "
    "sould always be the first item in this array of structures.",
    "radius": "float",
    "height": "float",
    "psi": "float",
    "levelset [structure]": {
        "doc": "Single poloidal flux contour incuding x-point.",
        "radius [:]": "float",
        "height [:]": "float",
    },
}
time_slice["o_point [array of structures]"] = {
    "doc": "Local extremum of the poloidal flux map. "
    "These points may be local minima or local maxima of the poloidal flux map. "
    "These points are rarely coincident with nodes defining the poloidal upon "
    "which the poloidal flux map is defined."
    "O-points are ordered first by proximitly to the magnetic axis, "
    "mesured by the number of edge conections required to link the point in question "
    "to the magnetic axis, and then by "
    "the value of poloidal flux multiplied by the sign of plasma current. A plasma "
    "current of zero is taken to be positive. For cases without a magnetic axis points "
    "are ordered soley on the value of poloidal flux. If present, the o-point "
    "describing the magnetic axis sould always be the first item in this array of "
    "structures.",
    "radius": "float",
    "height": "float",
    "psi": "float",
    "critical_type": "0-minimum, 2-maximum",
}
time_slice["contour_tree"] = tree

pprint(time_slice, indent=1, depth=50, compact=False, width=90, sort_dicts=False)


class tree:
    """
    Tree strucutre for storing."""
