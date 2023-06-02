from sympy import symbols, simplify, integrate, pretty, pi, cos


x, dx, xc, h, r, theta, f = symbols("x dx xc h r theta f", real=True, positive=True)

"""
square = 1/dx * integrate(x**2, (x, x-dx/2, x+dx/2))
print(pretty(simplify(square)))

circle = 4 / (pi * dx**2) * integrate(
        integrate(r * (xc + r * cos(theta))**2, (r, 0, dx/2)), 
        (theta, 0, 2*pi))
print(pretty(simplify(circle)))
"""

skin = (
    1
    / (pi * dx)
    * integrate(dx / 2 * (xc + dx / 2 * cos(theta)) ** 2, (theta, 0, 2 * pi))
)
print(pretty(simplify(skin)))

thick_skin = (4 / (pi * dx**2) - 4 / (pi * ((1 - f) * dx) ** 2)) * integrate(
    integrate(r * (xc + r * cos(theta)) ** 2, (r, (1 - f) * dx / 2, dx / 2)),
    (theta, 0, 2 * pi),
)
print(pretty(simplify(thick_skin)))
