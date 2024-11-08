"""Decompose elephantine shape."""

import pathlib

import appdirs
import numpy as np
import matplotlib.pyplot as plt

import scipy.fft

import PIL

path = pathlib.Path(appdirs.user_data_dir()) / "fair-mast"

Ax = np.array([-60, 0, 0, 0, 0, 0])
Bx = np.array([-30, 8, -10, 0, 0, 0])

Ay = -np.array([0, 0, 12, 0, -14, 0])
By = -np.array([50, 18, 0, 0, 0, 0])

Cx = (Ax - 1j * Bx) / 2
Cy = (Ay - 1j * By) / 2


point_number = 251
"""
t = np.linspace(0, 2 * np.pi, point_number)


Tn = np.array([0, 1, 1, -0.25])
points = np.zeros(point_number)
Wa = 10
points[:10] = Wa
points[-10:] = Wa

wiggle = 1j * points
Wn = scipy.fft.fft(wiggle)
"""
x = scipy.fft.irfft(np.r_[0, Cx], point_number)
y = scipy.fft.irfft(np.r_[0, Cy], point_number)

Cn = scipy.fft.fft(x + 1j * y)

coef_number = 12
# Cn = np.array([0, 1, 2, 3, -4, -3, -2, -1])
# Cn = np.r_[Cn[: coef_number // 2], Cn[-coef_number // 2 + 1 :]]

index = abs(Cn) < 0.5
Cn[index] = 0

Cn = np.r_[Cn[: coef_number // 2], Cn[-coef_number // 2 + 1 :]]

# five parameters to fit elephantine profile + eye (complex, so really 10)
Pn = np.array(
    [
        -55 + 15j,
        -9 - 4j,
        0 + 7j,
        -5 - 11j,
        20 + 1j,
    ]
)


def elephant_coefficents(Pn, point_number: int = 251):
    """Return complex fourier coefficents for elephantine profile."""
    Cn = np.zeros(point_number, dtype=complex)
    Cn[1:3] = Pn[:2]
    Cn[5] = Pn[2]
    Cn[-5] = Pn[2]
    Cn[-3:] = 1j * Pn[3].imag, -Pn[1], Pn[3].real - 1j * Pn[0].imag
    return Cn


t = np.linspace(0, 2 * np.pi, point_number)
mode_number = np.arange(-5, 6)
# profile = np.exp(-1j * n)


def elephant_profile(Pn, factor: float = 0.0, point_number: int = 251):
    """Return complex profile of elephantine shape."""
    # reference profile
    Cn = elephant_coefficents(Pn, point_number)
    # wiggle trunk
    if not np.isclose(factor, 0.0):
        Cn[abs(Cn) > 0] += 1j * factor * Pn[4].imag
    profile = point_number * scipy.fft.ifft(Cn, point_number)
    return np.append(profile, profile[0])


def plot_elephant(Pn, factor=0, axes=None):
    """Return outline and eye plot objexts for elephantine profile."""
    profile = elephant_profile(Pn, factor)
    if axes is None:
        axes = subplots()[1]
    return (
        axes.plot(profile.real, profile.imag, "-", color="gray")[0],
        axes.plot(Pn[4].real, Pn[4].real, "o", color="gray")[0],
    )


def decompose_elephant(Pn, n):
    """Return elephant profile split into 4 modes."""
    n = 500
    Cn = elephant_coefficents(Pn, 11)[:, np.newaxis]
    k = np.arange(-5, 6)[:, np.newaxis]
    return Cn[k[:, 0]] * np.exp(2j * np.pi * k * np.arange(n)[np.newaxis] / n)


def hello_elephant(Pn) -> PIL.Image.Image:
    """Yield elephant images with animated trunk."""
    fig, axes = subplots()
    outline = plot_elephant(Pn, axes=axes)[0]
    for factor in np.sin(np.linspace(0, 2 * np.pi, 30)):
        # update elephant outline
        profile = elephant_profile(Pn, factor)
        outline.set_data(profile.real, profile.imag)
        fig.canvas.draw()
        yield PIL.Image.fromarray(np.array(fig.canvas.buffer_rgba()))


def draw_elephant(Pn, step=6) -> PIL.Image.Image:
    """Yield partial elephant plot."""
    fig, axes = subplots()
    outline = plot_elephant(Pn, axes=axes)
    profile = elephant_profile(Pn)
    for i in np.arange(len(profile[::step]) + 1):
        outline.set_data(profile.real[: i * step], profile.imag[: i * step])
        fig.canvas.draw()
        yield PIL.Image.fromarray(np.array(fig.canvas.buffer_rgba()))


def save_gif(generator, filename: str, *args, duration=5):
    """Save output images from generator function to gif annimation."""
    images = [image for image in generator(*args)]
    images[0].save(
        path / filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def fourier_components(profile: np.ndarray, k=5) -> np.ndarray:
    """Yield Forier components for elephantine shape."""
    origin = 0 + 0j
    for mode in np.arange(1, len(profile) // 2 + 1):
        if mode > k:
            break
        for i in (1, -1):
            point = origin + profile[i * mode]
            yield [origin.real, point.real], [origin.imag, point.imag]
            origin = point


def plot_fourier_components(profile: np.ndarray, k=5, axes=None):
    """Plot Forier components for elephantine shape."""
    if axes is None:
        axes = subplots()[1]
    components = []
    for i, line in enumerate(fourier_components(profile, k)):
        components.append(axes.plot(*line, f"-C{i // 2}")[0])
    for i, component in enumerate(components[::2]):
        if i == 3:  # mode 4 is empty
            continue
        component.set_label(f"k={i+1}")
    axes.legend(loc="upper right")
    components.append(axes.plot(line[0][1], line[1][1], "ko", ms=8)[0])
    return components


def set_fourier_components(components, profile: np.ndarray, k=5):
    """Set data for for fourier components."""
    for component, line in zip(components, fourier_components(profile, k)):
        component.set_data(line)
    components[-1].set_data([line[0][1:2], line[1][1:2]])


def partial_elephant(profile, k=5):
    """Return partial elephant profile."""
    return np.sum(profile[: k + 1], axis=0) + np.sum(profile[-k:], axis=0)


def plot_partial_elephant(profile, k=5, axes=None):
    """Return outline for first k components of elephantine profile."""
    if axes is None:
        axes = subplots()[1]
    return axes.plot(profile.real, profile.imag, "-", color="gray")[0]


def set_partial_elephant(outline, profile):
    """Set data for partial elephant profile."""
    outline.set_data(profile.real, profile.imag)


def draw_partial_elephant(Pn, k=5, skip=50, eye=False):
    """Yield partial elephant frames."""
    fig, axes = subplots()
    if eye:
        axes.plot(Pn[4].real, Pn[4].real, "o", color="gray")
    profile = scipy.fft.ifftshift(decompose_elephant(Pn, 500), 0)
    plot_partial_elephant(partial_elephant(profile, k), k, axes=axes)
    components = plot_fourier_components(profile[:, 0], k, axes=axes)
    for n in np.arange(profile.shape[1])[::skip]:
        set_fourier_components(components, profile[:, n], k)
        fig.canvas.draw()
        yield PIL.Image.fromarray(np.array(fig.canvas.buffer_rgba()))


def subplots():
    """Return fig, axes."""
    fig, axes = plt.subplots(figsize=(4, 4))
    axes.set_aspect("equal")
    axes.set_axis_off()
    axes.set_xlim(-80, 100)
    axes.set_ylim(-80, 100)
    fig.tight_layout(pad=0)
    return fig, axes


# outline = plot_elephant(Pn, axes=axes)[0]
# for k in [1, 2, 3, 5]:
#    save_gif(draw_partial_elephant, f"partial_elephant_k{k}.gif", Pn, k, 10)
save_gif(hello_elephant, "elephant_trunk.gif", Pn)
# save_gif(draw_elephant, "elephant_sketch.gif", Pn, outline)
# save_gif(draw_partial_elephant, "partial_elephant_k5_eye.gif", Pn, 5, 10, True)
k = 5

fig, axes = subplots()
profile = scipy.fft.ifftshift(decompose_elephant(Pn, 500), 0)
plot_partial_elephant(partial_elephant(profile, k), k, axes=axes)
components = plot_fourier_components(profile[:, 0], k, axes=axes)


# IPython.display.Image(open("elephant_trunk.gif", "rb").read())

# profile = elephant_profile(Pn)
# axes.plot(profile.real, profile.imag, "-", color="gray")

# plot_elephant(Pn, axes=axes)[0]


"""

coef = np.array(
    [
        50 - 30j,
        18 + 8j,
        -10 + 12j,
        -14 - 60j,
        # 40 + 20j,
    ]
)

Cn = np.r_[0, coef]

profile = scipy.fft.ifft(Cn, 251)
plt.plot(profile.real, profile.imag)

N = 250
t = np.linspace(0, 2 * np.pi, 250)[np.newaxis]
profile = (
    Cn[:, np.newaxis]
    / 250
    * np.exp(1j * np.arange(0, len(coef) + 1)[:, np.newaxis] * t)
).sum(axis=0)

plt.figure()
plt.plot(profile.real, profile.imag)
"""

"""
t = np.linspace(0, 2 * np.pi, 251)
plt.plot(
    -60 * np.cos(t) + 30 * np.sin(t) - 8 * np.sin(2 * t) + 10 * np.sin(3 * t),
    50 * np.sin(t) + 18 * np.sin(2 * t) - 12 * np.cos(3 * t) + 14 * np.cos(5 * t),
)
"""
