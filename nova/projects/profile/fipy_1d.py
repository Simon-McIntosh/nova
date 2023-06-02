import fipy

D = 0.1
steps = 20
nx = 50
dx = 1.0

mesh = fipy.Grid1D(nx=nx, dx=dx)
x = mesh.cellCenters[0]

timeStepDuration = 0.9 * dx**2 / (2 * D)
t = timeStepDuration * steps

phi = fipy.CellVariable(name="solution variable", mesh=mesh, value=0.0)
print(mesh.facesRight)
phi.constrain(1, mesh.facesRight)
phi.constrain(0.5, mesh.facesLeft)

viewer = fipy.Viewer(vars=(phi), datamin=0.0, datamax=1.0)
viewer.plot()

"""
eqX = fipy.TransientTerm() == fipy.ExplicitDiffusionTerm(coeff=D)
for step in range(steps):
    eqX.solve(var=phi, dt=timeStepDuration)

phiAnalytical = fipy.CellVariable(name="analytical value", mesh=mesh)
phiAnalytical.setValue(1 - erf(x / (2 * fipy.numerix.sqrt(D * t))))


viewer = fipy.Viewer(vars=(phi), datamin=0.0, datamax=1.0)
#viewer.plot()
"""
