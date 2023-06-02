import fipy

cellSize = 0.05
radius = 1.0

msh_txt = """
cellSize = 0.05;
radius = 1.0;
Point(1) = {0, 0, 0, cellSize};
Point(2) = {-radius, 0, 0, cellSize};
Point(3) = {0, radius, 0, cellSize};
Point(4) = {radius, 0, 0, cellSize};
Point(5) = {0, -radius, 0, cellSize};
Circle(6) = {2, 1, 3};
Circle(7) = {3, 1, 4};
Circle(8) = {4, 1, 5};
Circle(9) = {5, 1, 2};
Line Loop(10) = {6, 7, 8, 9};
Plane Surface(11) = {10};
"""

print(msh_txt)

mesh = fipy.Gmsh2D(msh_txt % locals())
