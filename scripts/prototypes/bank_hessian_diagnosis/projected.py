"""Probe a rotated rectangle near the extension of a source edge."""

# ruff: noqa: E402
import json

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

from nova.biot.greens import MU0
from nova.equilibrium.topology import BiotMomentCoupling
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT, reference
from scripts.prototypes.read_route.jets import _source_jets


def boundary(point, polygon, digits):
    with mp.workdps(digits):
        r, z = [mp.mpf(float(v)) for v in point]
        vertices = [[mp.mpf(float(v)) for v in p] for p in polygon]

        def kernel(radius, height, a, b):
            parameter = 4 * radius * a / ((radius + a) ** 2 + (height - b) ** 2)
            return (
                mp.mpf(float(MU0))
                * mp.sqrt(radius * a / parameter)
                * ((2 - parameter) * mp.ellipk(parameter) - 2 * mp.ellipe(parameter))
            )

        def vertical(radius, height):
            value = mp.mpf(0)
            for i, a in enumerate(vertices):
                b = vertices[(i + 1) % len(vertices)]
                dr, dz = b[0] - a[0], b[1] - a[1]
                value += dr * mp.quad(
                    lambda t: kernel(radius, height, a[0] + t * dr, a[1] + t * dz),
                    [0, 1],
                )
            return 200000 * value

        values = [
            mp.diff(lambda radius: vertical(radius, z), r),
            mp.diff(lambda height: vertical(r, height), z),
        ]
        return np.array(
            [mp.nstr(value, digits - 5) for value in values], dtype=np.longdouble
        )


def main():
    assert jax.config.jax_enable_x64
    angle = np.pi / 6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    polygon = np.array(
        [[-0.5, -0.6], [0.5, -0.6], [0.5, 0.6], [-0.5, 0.6]]
    ) @ rotation.T + [6, 0]
    edge = polygon[2] - polygon[1]
    normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
    projected = polygon[1] + 2 * edge
    coupling = BiotMomentCoupling.from_polygons((polygon,))
    ref = reference()
    packed = ref.coupling((polygon,))
    moments = np.array([[240000.0, 0.0, 0.0]])
    output = []
    for distance in (1e-12, 1e-4):
        point = projected + distance * normal
        truth = boundary(point, polygon, 35)
        refined = boundary(point, polygon, 50)
        item = dict(
            point=point.tolist(),
            polygon=polygon.tolist(),
            distance_m=distance,
            independent=refined.tolist(),
            precision_change=abs(truth - refined).tolist(),
        )
        (ROOT / "projected-partial.json").write_text(
            json.dumps(item, indent=2, default=float)
        )
        print("PROJECTED_INDEPENDENT", json.dumps(item, default=float), flush=True)
        bank = np.asarray(_source_jets(jnp.array(point), coupling))[0, 0] * 240000
        derivatives = []
        for step in (0.001, 0.0005, 0.00025):
            shift = np.array([0, step], dtype=np.longdouble)
            values = ref.gradient(
                np.array([point + shift, point - shift]), packed, moments
            )
            derivatives.append((values[0] - values[1]) / (2 * step))
        fourth = (4 * np.array(derivatives[1:]) - np.array(derivatives[:-1])) / 3
        finite = (16 * fourth[1] - fourth[0]) / 15
        item.update(bank=bank.tolist(), reference=finite.tolist())
        output.append(item)
        (ROOT / "projected.json").write_text(
            json.dumps(output, indent=2, default=float)
        )
        print("PROJECTED", json.dumps(item, default=float), flush=True)
    print("PROJECTED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
