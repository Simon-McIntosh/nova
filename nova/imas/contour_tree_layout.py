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
    "doc": "A structure to store the location, "
    "value, and connectivity of poloidal flux critical points."
}
tree["node [structure]"] = {
    "critical_type [:]": "0-minimum, 1-saddle, 2-maximum",
    "doc": "For saddle type geometries (critical type = 1) the point index "
    "attribute references items from the x_point structure otherwise items are "
    "referenced from the o_point structure.",
    "point_index [:]": "float",
}
tree["edge [structure"] = "[int, int]"

time_slice = {}
time_slice["x_point [array of structures]"] = {
    "doc": "Points are ordered first by proximitly to the magnetic axis "
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
    "doc": "Points are ordered first by proximitly to the magnetic axis "
    ",mesured by the number of edge conections required to link the point in question "
    "to the magnetic axis, and then by "
    "the value of poloidal flux multiplied by the sign of plasma current. A plasma "
    "current of zero is taken to be positive. For cases without a magnetic axis points "
    "are ordered soley on the value of poloidal flux. If present, the o-point "
    "describing the magnetic axis sould always be the first item in this array of "
    "structures.",
    "radius [:]": "float",
    "height [:]": "float",
    "psi [:]": "float",
    "critical_type [:]": "0-minimum, 2-maximum",
}
time_slice["contour_tree"] = tree

pprint(time_slice, indent=1, depth=50, compact=False, width=120, sort_dicts=False)


class tree:
    """
    Tree strucutre for storing."""
