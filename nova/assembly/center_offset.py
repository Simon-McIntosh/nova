import numpy as np
import scipy

import matplotlib.pyplot as plt

phi = np.linspace(0, 2 * np.pi, 300)

radius = 4.1053

n_filament = 1

if n_filament == 1:
    delta = np.array([[-0.01957680486843255], [-0.01463439518910984]])
elif n_filament == 2:
    delta = np.array([[-0.0025, -0.0025], [-0.0025, 0.0025]])
elif n_filament == 3:
    delta = np.array(
        [
            [-0.025, -0.025, -0.02, -0.02, -0.02, -0.02],
            [-0.025, 0.025, -0.025, -0.025, -0.025, -0.025],
        ]
    )

r_hat = np.array([np.cos(phi), np.sin(phi)])
t_hat = np.array([-np.sin(phi), np.cos(phi)])

Bo, Br, Bphi = 0, 0, 0
for dx, dy in zip(*delta):
    r_dash = np.array([radius * np.cos(phi) - dx, radius * np.sin(phi) - dy])
    r_dash_norm = np.linalg.norm(r_dash, axis=0)
    t_dash = np.array([-r_dash[1], r_dash[0]])
    t_dash_hat = t_dash / np.linalg.norm(t_dash, axis=0)

    Bo += 1 / radius
    Br += 1 / r_dash_norm * np.einsum("ij,ij->j", t_dash_hat, r_hat)
    Bphi += 1 / r_dash_norm * np.einsum("ij,ij->j", t_dash_hat, t_hat)
Bphi -= Bo

offset_zt, offset_zr = np.zeros(2), np.zeros(2)
offset_zt[0] = radius / (np.pi * Bo) * np.trapz(Bphi * np.cos(phi), phi)
offset_zt[1] = radius / (np.pi * Bo) * np.trapz(Bphi * np.sin(phi), phi)

offset_zr[0] = -radius / (np.pi * Bo) * np.trapz(Br * np.sin(phi), phi)
offset_zr[1] = radius / (np.pi * Bo) * np.trapz(Br * np.cos(phi), phi)

h = radius * scipy.integrate.cumulative_trapezoid(Br / Bo, phi, initial=0)

offset_h = np.fft.rfft(h)[1] / (len(h) // 2)


_delta = scipy.interpolate.interp1d(phi, np.array([Br, Bphi]))


def fun(t, y, radial_only=False):
    phi = np.arctan2(y[1], y[0])
    if phi < 0:
        phi += 2 * np.pi
    delta = _delta(phi)
    if radial_only:
        Bx = delta[0] * np.cos(phi) - np.sin(phi)
        By = delta[0] * np.sin(phi) + np.cos(phi)
        return Bx, By
    Bx = delta[0] * np.cos(phi) - delta[1] * np.sin(phi) - Bo * np.sin(phi)
    By = delta[0] * np.sin(phi) + delta[1] * np.cos(phi) + Bo * np.cos(phi)
    return Bx, By


def loop(t, y):
    if t == 0:
        return 1
    return y[1]


loop.terminal = True
loop.direction = 1

t_max = 10 * np.pi * radius
sol = scipy.integrate.solve_ivp(
    fun,
    (0, t_max),
    (radius, 0),
    t_eval=np.linspace(0, t_max, 200),
    rtol=1e-5,
    events=loop,
    max_step=0.5,
)

phi_sol = np.arctan2(sol.y[1], sol.y[0])
phi_sol[phi_sol < 0] += 2 * np.pi
circle = np.array([radius * np.cos(phi_sol), radius * np.sin(phi_sol)])
delta_sol = sol.y - circle
h_sol = np.linalg.norm(delta_sol, axis=0)
h_sol = np.interp(phi, phi_sol, h_sol)

offset_sol = np.fft.rfft(h_sol)[1] / (len(h_sol) // 2)

factor = len(delta) * 100

axes = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[2, 1]))[1]

axes[0].plot(
    radius * np.cos(phi), radius * np.sin(phi), "-.", color="gray", label="wall"
)

axes[0].plot(
    radius * np.cos(phi) + factor * h * np.cos(phi),
    radius * np.sin(phi) + factor * h * np.sin(phi),
    "-",
    color="C2",
    label=r"$h(\phi)$",
)
axes[0].plot(
    radius * np.cos(phi) + factor * offset_zt[0],
    radius * np.sin(phi) + factor * offset_zt[1],
    "-.",
    color="C2",
    label=r"$\zeta_t$ wall",
)
axes[0].plot(
    radius * np.cos(phi) + factor * offset_zr[0],
    radius * np.sin(phi) + factor * offset_zr[1],
    "-.",
    color="C2",
    label=r"$\zeta_r$ wall",
)

axes[0].plot(
    circle[0] + factor * delta_sol[0],
    circle[1] + factor * delta_sol[1],
    "C1--",
    label="RK45(B)",
)
axes[0].plot(
    radius * np.cos(phi) + factor * offset_sol.real,
    radius * np.sin(phi) - factor * offset_sol.imag,
    "-.",
    color="C1",
    label="RK45 wall",
)

axes[0].plot(0, 0, "X", color="gray")
axes[0].plot(factor * delta[0], factor * delta[1], "k.")

axes[0].axis("equal")
axes[0].axis("off")
axes[0].legend(fontsize="small", bbox_to_anchor=(0.71, 1))

axes[1].plot(phi, 1e3 * h, "C2")
axes[1].plot(phi, 1e3 * h_sol, "C1--")
axes[1].set_xlabel(r"$\phi$")
axes[1].set_ylabel(f"$h$, mm")
plt.despine()

"""
plt.figure()
plt.plot(phi, Br)
plt.plot(phi, Bphi)
plt.plot(phi, h)
"""
print(offset_zt)
print(offset_zr)
print(offset_sol.real, -offset_sol.imag)
# print(np.abs(offset), np.linalg.norm(offset_zt), np.linalg.norm(offset_zr))
