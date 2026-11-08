"""Signed residual colours and their full-severity colour-vision check."""

from itertools import combinations

from matplotlib.colors import to_rgb
import numpy as np

POSITIVE = "#2e8b57"
NEGATIVE = "#7954ad"
FLUX = "#929292"
BOUNDARY = "#666666"

# Machado, Oliveira and Fernandes: doi:10.1109/TVCG.2009.113,
# severity 100 matrices, applied to linear sRGB before the D65 Lab conversion.
VISION = {
    "normal": np.eye(3),
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}


def lab(colour, vision):
    rgb = np.asarray(to_rgb(colour))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    simulated = np.clip(vision @ linear, 0, 1)
    xyz = (
        np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
        @ simulated
        / np.array([0.95047, 1, 1.08883])
    )
    delta = 6 / 29
    transformed = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    x, y, z = transformed
    return np.array([116 * y - 16, 500 * (x - y), 200 * (y - z)])


def validate_palette():
    """Require CIE76 delta-E at least eight for both signs and every flux grey."""
    assert (
        np.linalg.norm(lab("white", VISION["normal"]) - lab("black", VISION["normal"]))
        > 99
    )
    assert np.array_equal(
        lab(POSITIVE, VISION["normal"]), lab(POSITIVE, VISION["normal"])
    )
    pairs = [
        pair
        for pair in combinations((POSITIVE, NEGATIVE, FLUX, BOUNDARY), 2)
        if pair != (FLUX, BOUNDARY)
    ]
    for name, matrix in VISION.items():
        distances = [np.linalg.norm(lab(a, matrix) - lab(b, matrix)) for a, b in pairs]
        assert min(distances) >= 8, (name, distances)
        print(f"PALETTE {name}: minimum CIE76 delta-E={min(distances):.3f}; required=8")


if __name__ == "__main__":
    validate_palette()
