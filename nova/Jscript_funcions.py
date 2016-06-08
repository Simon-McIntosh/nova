
def set_plane(f):
    f.write('//Plane\n')
    f.write('p.Plane  = agb.GetActivePlane();\n')
    f.write('p.Origin = p.Plane.GetOrigin();\n')
    f.write('p.XAxis  = p.Plane.GetXAxis();\n')
    f.write('p.YAxis  = p.Plane.GetYAxis();\n')
    f.write('\n')