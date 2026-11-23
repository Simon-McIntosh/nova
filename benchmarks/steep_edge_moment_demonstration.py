# ruff: noqa: E501
"""Demonstrate coarse plasma-current moments against analytic references.

The source domain is the exact static member of the rotating-equilibrium
reference family.  Its Solov'ev current is exercised directly and two
logistic edge pedestals sharpen the same current as a function of normalised
flux.  Fixed rectangular source sections use all nine exact polygon blocks.
Their three changing vectors come from the degree-three interior stencil and
the conservative any-intersection boundary clip.

The independent reference integrates point-filament kernels in analytic
flux coordinates.  A nested quadrature records its own convergence.  For the
static member, the converged plasma response is completed by the exact
homogeneous contribution, recovering the closed-form Solov'ev flux and both
field components at every target.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable

import numpy as np

from nova.biot.greens import greens_bz_br, greens_psi
from nova.biot.polygonanalytic import (
    polygon_analytic_field_moments,
    polygon_analytic_flux_moments,
)
from nova.equilibrium.separatrix_clip import (
    AtomicCellMesh,
    padded_linear_current_moments,
)
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import reference_cases

DEFAULT_OUTPUT = Path("docs/figures/plasma-edge-current-representation")
PITCHES = (0.11, 0.08, 0.06)
TARGET_COUNT = 48
REFERENCE_ORDERS = ((40, 80), (64, 128))
PROFILE_SPECS = (
    ("solovev_exact", None),
    ("pedestal_moderate", 0.10),
    ("pedestal_steep", 0.035),
)
PEDESTAL_CENTRE = 0.86
PEDESTAL_AMPLITUDE = 1.5
AREA_TOLERANCE = 1.0e-12


def _rectangles(centres: np.ndarray, pitch: float) -> list[np.ndarray]:
    half = 0.5 * pitch
    offset = np.asarray([[-half, -half], [half, -half], [half, half], [-half, half]])
    return [centre + offset for centre in centres]


def _mesh(case, pitch: float) -> tuple[StencilMesh, list[np.ndarray], np.ndarray]:
    inner, outer = case.boundary_midplane_radii()
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    r_edge = np.arange(
        math.floor((inner - pitch) / pitch) * pitch,
        math.ceil((outer + pitch) / pitch) * pitch + 0.5 * pitch,
        pitch,
    )
    z_edge = np.arange(
        math.floor((-half_height - pitch) / pitch) * pitch,
        math.ceil((half_height + pitch) / pitch) * pitch + 0.5 * pitch,
        pitch,
    )
    radius = 0.5 * (r_edge[:-1] + r_edge[1:])
    height = 0.5 * (z_edge[:-1] + z_edge[1:])
    rr, zz = np.meshgrid(radius, height)
    coordinate = np.column_stack((rr.ravel(), zz.ravel()))
    nr = len(radius)
    nz = len(height)
    rings = []
    for iz in range(1, nz - 1):
        for ir in range(1, nr - 1):
            centre = iz * nr + ir
            rings.append(
                [
                    centre,
                    *(
                        (iz + dz) * nr + ir + dr
                        for dz, dr in (
                            (-1, -1),
                            (-1, 0),
                            (-1, 1),
                            (0, -1),
                            (0, 1),
                            (1, -1),
                            (1, 0),
                            (1, 1),
                        )
                    ),
                ]
            )
    mesh = StencilMesh(
        coordinate=coordinate,
        stencil=np.asarray(rings, dtype=np.intp),
        area=np.full(len(coordinate), pitch**2),
    )
    cells = _rectangles(coordinate, pitch)
    node_r, node_z = np.meshgrid(r_edge, z_edge)
    shared = np.column_stack((node_r.ravel(), node_z.ravel()))
    cell_node = np.empty((len(coordinate), 4), dtype=np.intp)
    for iz in range(nz):
        for ir in range(nr):
            cell = iz * nr + ir
            lower = iz * len(r_edge) + ir
            cell_node[cell] = (
                lower,
                lower + 1,
                lower + 1 + len(r_edge),
                lower + len(r_edge),
            )
    return mesh, cells, np.asarray((shared, cell_node), dtype=object)


def _normalised_flux(case, radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(case.flux(radius, height)) / case.axis_flux


def _density_function(case, width: float | None) -> Callable:
    def density(radius, height):
        base = np.asarray(case.toroidal_current_density(radius, height), dtype=float)
        if width is None:
            return base
        normalised = _normalised_flux(case, np.asarray(radius), np.asarray(height))
        argument = np.clip((normalised - PEDESTAL_CENTRE) / width, -40.0, 40.0)
        pedestal = 1.0 / (1.0 + np.exp(-argument))
        return base * (1.0 + PEDESTAL_AMPLITUDE * pedestal)

    return density


def _targets(case) -> np.ndarray:
    angle = 2.0 * np.pi * np.arange(TARGET_COUNT) / TARGET_COUNT
    half_u = math.sqrt(2.0 * case.axis_flux / case.pressure_coefficient)
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    radial_scale = 1.18
    vertical_scale = 1.30
    radius = np.sqrt(case.major_radius**2 + radial_scale * half_u * np.cos(angle))
    height = vertical_scale * half_height * np.sin(angle)
    return np.column_stack((radius, height))


def _reference_geometry(case, radial_order: int, angular_order: int):
    radial_node, radial_weight = np.polynomial.legendre.leggauss(radial_order)
    angular_node, angular_weight = np.polynomial.legendre.leggauss(angular_order)
    rho = 0.5 * (radial_node + 1.0)
    rho_weight = 0.5 * radial_weight
    angle = np.pi * (angular_node + 1.0)
    angle_weight = np.pi * angular_weight
    half_u = math.sqrt(2.0 * case.axis_flux / case.pressure_coefficient)
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    rr, aa = np.meshgrid(rho, angle, indexing="ij")
    source_r = np.sqrt(case.major_radius**2 + half_u * rr * np.cos(aa))
    source_z = half_height * rr * np.sin(aa)
    jacobian = half_u * half_height * rr / (2.0 * source_r)
    weight = jacobian * rho_weight[:, None] * angle_weight[None, :]
    return source_r.ravel(), source_z.ravel(), weight.ravel()


def _integrated_reference(case, targets: np.ndarray, order: tuple[int, int]):
    source_r, source_z, weight = _reference_geometry(case, *order)
    kernels = []
    for start in range(0, len(source_r), 4096):
        stop = min(start + 4096, len(source_r))
        sr = source_r[start:stop][None, :]
        sz = source_z[start:stop][None, :]
        tr = targets[:, 0, None]
        tz = targets[:, 1, None]
        psi = greens_psi(tr, tz, sr, sz)
        bz, br = greens_bz_br(tr, tz, sr, sz)
        kernels.append((psi, br, bz, weight[start:stop]))
    result = {}
    for name, width in PROFILE_SPECS:
        density = _density_function(case, width)
        response = np.zeros((3, len(targets)))
        offset = 0
        for psi, br, bz, local_weight in kernels:
            count = len(local_weight)
            local_density = density(
                source_r[offset : offset + count], source_z[offset : offset + count]
            )
            weighted = local_density * local_weight
            response[0] += psi @ weighted
            response[1] += br @ weighted
            response[2] += bz @ weighted
            offset += count
        result[name] = response
    return result


def _exact_solovev(case, targets: np.ndarray) -> np.ndarray:
    radius = targets[:, 0]
    height = targets[:, 1]
    label = radius**2 - case.major_radius**2
    flux_per_radian = np.asarray(case.flux(radius, height))
    return np.asarray(
        [
            2.0 * np.pi * flux_per_radian,
            2.0 * case.field_coefficient * height / radius,
            -2.0 * case.pressure_coefficient * label,
        ]
    )


def _blocks(targets: np.ndarray, mesh: StencilMesh, cells: list[np.ndarray]):
    block = np.empty((3, 3, len(targets), mesh.node_count))
    for source, vertices in enumerate(cells):
        flux = polygon_analytic_flux_moments(
            targets[:, 0],
            targets[:, 1],
            vertices,
            expansion_point=mesh.coordinate[source],
        )
        radial, vertical = polygon_analytic_field_moments(
            targets[:, 0],
            targets[:, 1],
            vertices,
            expansion_point=mesh.coordinate[source],
        )
        block[0, :, :, source] = flux
        block[1, :, :, source] = radial
        block[2, :, :, source] = vertical
    return block


def _vectors(mesh, clipped, shared, cell_node, density, pitch):
    centroid_density = density(mesh.coordinate[:, 0], mesh.coordinate[:, 1])
    shared_density = density(shared[:, 0], shared[:, 1])
    stencil = mesh.current_moment_stencil(
        cell_node, np.full((mesh.node_count, 2), pitch**2 / 12.0)
    )
    interior = stencil(centroid_density, shared_density)
    gradient = np.column_stack(mesh.gradient(centroid_density))
    boundary = clipped.linear_current_moments(centroid_density, gradient)
    current = np.where(
        clipped.boundary, boundary.current, np.asarray(interior.cell_current)
    )
    radial = np.where(
        clipped.boundary,
        12.0 * boundary.radial / pitch**2,
        12.0 * np.asarray(interior.radial_moment) / pitch**2,
    )
    vertical = np.where(
        clipped.boundary,
        12.0 * boundary.vertical / pitch**2,
        12.0 * np.asarray(interior.vertical_moment) / pitch**2,
    )
    admitted = clipped.included
    return (
        (
            np.where(admitted, current, 0.0),
            np.where(admitted, radial, 0.0),
            np.where(admitted, vertical, 0.0),
        ),
        centroid_density,
        gradient,
    )


def _contract(block: np.ndarray, vectors: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.asarray(
        [sum(block[q, m] @ vectors[m] for m in range(3)) for q in range(3)]
    )


def _metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    names = ("psi", "b_r", "b_z")
    result = {}
    for index, name in enumerate(names):
        span = float(np.ptp(reference[index]))
        difference = actual[index] - reference[index]
        result[name] = {
            "reference_span": span,
            "sup_fraction_of_span": float(np.max(np.abs(difference)) / span),
            "rms_fraction_of_span": float(np.sqrt(np.mean(difference**2)) / span),
        }
    return result


def _fit_orders(rows: list[dict[str, Any]], representation: str):
    fitted = {}
    pitch = np.asarray([row["pitch_m"] for row in rows])
    for quantity in ("psi", "b_r", "b_z"):
        fitted[quantity] = {}
        for norm in ("sup_fraction_of_span", "rms_fraction_of_span"):
            error = np.asarray([row[representation][quantity][norm] for row in rows])
            slope, intercept = np.polyfit(np.log(pitch), np.log(error), 1)
            predicted = slope * np.log(pitch) + intercept
            residual = np.sum((np.log(error) - predicted) ** 2)
            total = np.sum((np.log(error) - np.mean(np.log(error))) ** 2)
            fitted[quantity][norm] = {
                "order": float(slope),
                "coefficient": float(np.exp(intercept)),
                "r_squared": float(1.0 - residual / total) if total else 1.0,
            }
    return fitted


def _cpu_timing(iterations, atomic, density, gradient):
    import jax
    import jax.numpy as jnp

    configure_dtypes()
    trace_count = 0

    def contract(vertices, count, centroids):
        nonlocal trace_count
        trace_count += 1
        return padded_linear_current_moments(
            vertices, count, centroids, density, gradient
        )

    cpu = jax.devices("cpu")[0]
    compiled = jax.jit(contract)
    times = []
    shapes = set()
    for clipped in iterations:
        shapes.add((clipped.support_vertices.shape, clipped.vertex_count.shape))
        started = time.perf_counter()
        value = compiled(
            jax.device_put(jnp.asarray(clipped.support_vertices), cpu),
            jax.device_put(jnp.asarray(clipped.vertex_count), cpu),
            jax.device_put(jnp.asarray(atomic.centroids), cpu),
        )
        jax.block_until_ready(value)
        times.append(time.perf_counter() - started)
    return {
        "backend": "jax-cpu",
        "trace_count": trace_count,
        "shape_count": len(shapes),
        "first_iteration_seconds": float(times[0]),
        "warm_mean_seconds": float(np.mean(times[1:])),
        "warm_median_seconds": float(np.median(times[1:])),
    }


CUDA_SOURCE = r"""#include <cuda_runtime.h>
#include <chrono>
extern "C" __global__ void moments(const double* v,const long* n,const double* c,
 const double* d,const double* g,int cells,int cap,double* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=cells)return;
 long count=n[i]; double a2=0,fr=0,fz=0,rr=0,zz=0,rz=0;
 for(int k=0;k<count;k++){int q=(k+1<count)?k+1:0;
  double x=v[(i*cap+k)*2]-c[i*2], y=v[(i*cap+k)*2+1]-c[i*2+1];
  double X=v[(i*cap+q)*2]-c[i*2], Y=v[(i*cap+q)*2+1]-c[i*2+1];
  double cr=x*Y-X*y; a2+=cr; fr+=(x+X)*cr/6.; fz+=(y+Y)*cr/6.;
  rr+=(x*x+x*X+X*X)*cr/12.; zz+=(y*y+y*Y+Y*Y)*cr/12.;
  rz+=(2*x*y+x*Y+X*y+2*X*Y)*cr/24.;}
 double s=a2<0?-1.:1.; double area=.5*s*a2; fr*=s; fz*=s; rr*=s; zz*=s; rz*=s;
 out[i*3]=d[i]*area+g[i*2]*fr+g[i*2+1]*fz;
 out[i*3+1]=d[i]*fr+rr*g[i*2]+rz*g[i*2+1];
 out[i*3+2]=d[i]*fz+rz*g[i*2]+zz*g[i*2+1];}
extern "C" int run(const double* allv,const long* alln,const double* c,const double* d,
 const double* g,int iters,int cells,int cap,double* out,double* elapsed){
 double *v,*dc,*dd,*dg,*o; long *n; size_t vs=(size_t)cells*cap*2*sizeof(double);
 if(cudaMalloc(&v,vs)||cudaMalloc(&n,cells*sizeof(long))||cudaMalloc(&dc,cells*2*sizeof(double))||
 cudaMalloc(&dd,cells*sizeof(double))||cudaMalloc(&dg,cells*2*sizeof(double))||cudaMalloc(&o,cells*3*sizeof(double)))return 1;
 cudaMemcpy(dc,c,cells*2*sizeof(double),cudaMemcpyHostToDevice); cudaMemcpy(dd,d,cells*sizeof(double),cudaMemcpyHostToDevice);
 cudaMemcpy(dg,g,cells*2*sizeof(double),cudaMemcpyHostToDevice); auto start=std::chrono::steady_clock::now();
 for(int j=0;j<iters;j++){cudaMemcpy(v,allv+(size_t)j*cells*cap*2,vs,cudaMemcpyHostToDevice);
 cudaMemcpy(n,alln+(size_t)j*cells,cells*sizeof(long),cudaMemcpyHostToDevice);
 moments<<<(cells+255)/256,256>>>(v,n,dc,dd,dg,cells,cap,o); cudaMemcpy(out,o,cells*3*sizeof(double),cudaMemcpyDeviceToHost);}
 cudaDeviceSynchronize(); auto stop=std::chrono::steady_clock::now();
 *elapsed=std::chrono::duration<double>(stop-start).count(); int e=(int)cudaGetLastError();
 cudaFree(v);cudaFree(n);cudaFree(dc);cudaFree(dd);cudaFree(dg);cudaFree(o);return e;}
"""


def _cuda_timing(iterations, atomic, density, gradient):
    vertices = np.ascontiguousarray(
        np.stack([item.support_vertices for item in iterations]), dtype=np.float64
    )
    counts = np.ascontiguousarray(
        np.stack([item.vertex_count for item in iterations]), dtype=np.int64
    )
    centres = np.ascontiguousarray(atomic.centroids, dtype=np.float64)
    density = np.ascontiguousarray(density, dtype=np.float64)
    gradient = np.ascontiguousarray(gradient, dtype=np.float64)
    output = np.empty((len(centres), 3), dtype=np.float64)
    elapsed = ctypes.c_double()
    with tempfile.TemporaryDirectory(prefix="nova-edge-cuda-") as directory:
        source = Path(directory) / "moments.cu"
        library = Path(directory) / "moments.so"
        source.write_text(CUDA_SOURCE)
        completed = subprocess.run(
            [
                "/usr/local/cuda/bin/nvcc",
                "-O3",
                "-ccbin",
                "/usr/bin/g++",
                "--shared",
                "-Xcompiler",
                "-fPIC",
                str(source),
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        del completed
        run = ctypes.CDLL(str(library)).run
        pointer = ctypes.POINTER(ctypes.c_double)
        run.argtypes = [
            pointer,
            ctypes.POINTER(ctypes.c_long),
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            pointer,
        ]
        run.restype = ctypes.c_int
        arguments = (
            vertices.ctypes.data_as(pointer),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
            centres.ctypes.data_as(pointer),
            density.ctypes.data_as(pointer),
            gradient.ctypes.data_as(pointer),
            len(iterations),
            len(centres),
            vertices.shape[2],
            output.ctypes.data_as(pointer),
            ctypes.byref(elapsed),
        )
        status = run(*arguments)
        if status:
            raise RuntimeError(f"CUDA moment contraction failed with status {status}")
        status = run(*arguments)
        if status:
            raise RuntimeError(f"CUDA timed contraction failed with status {status}")
    expected = iterations[-1].linear_current_moments(density, gradient)
    difference = max(
        float(np.max(np.abs(output[:, 0] - expected.current))),
        float(np.max(np.abs(output[:, 1:] - expected.first))),
    )
    name = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.splitlines()[0]
        .strip()
    )
    return {
        "backend": "cuda-c-kernel",
        "device": name,
        "includes_host_device_transfers": True,
        "warm_mean_seconds": float(elapsed.value / len(iterations)),
        "host_agreement_sup": difference,
    }


def _iteration_timing(case, mesh, cells, density, gradient):
    atomic = AtomicCellMesh.from_cells(cells, centroids=mesh.coordinate)
    iterations = []
    for displacement in np.linspace(-0.012, 0.012, 12):
        level = atomic.sample(case.flux) - displacement * case.axis_flux
        iterations.append(atomic.clip(level))
    cpu = _cpu_timing(iterations, atomic, density, gradient)
    try:
        cuda = _cuda_timing(iterations, atomic, density, gradient)
    except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError) as error:
        cuda = {"backend": "unavailable", "reason": str(error)}
    return {"iterations": len(iterations), "cpu": cpu, "cuda": cuda}


def _plot(findings: dict[str, Any], output: Path):
    panels = ((70, 55), (555, 55), (70, 405), (555, 405))
    panel_width, panel_height = 420, 260
    colours = ("#1469a8", "#d97706", "#16803c")
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1050 750" role="img" aria-labelledby="title description">',
        '<title id="title">Coarse-mesh plasma current convergence</title>',
        '<desc id="description">Source profiles and log-log errors for flux and both field components.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    def draw_axes(x, y, title, xlabel, ylabel):
        lines.extend(
            [
                f'<text x="{x + panel_width / 2}" y="{y - 20}" text-anchor="middle" font-size="17">{title}</text>',
                f'<line x1="{x}" y1="{y + panel_height}" x2="{x + panel_width}" y2="{y + panel_height}" stroke="#333"/>',
                f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + panel_height}" stroke="#333"/>',
                f'<text x="{x + panel_width / 2}" y="{y + panel_height + 38}" text-anchor="middle" font-size="13">{xlabel}</text>',
                f'<text x="{x - 48}" y="{y + panel_height / 2}" text-anchor="middle" font-size="13" transform="rotate(-90 {x - 48} {y + panel_height / 2})">{ylabel}</text>',
            ]
        )

    x, y = panels[0]
    draw_axes(x, y, "Analytic source profiles", "normalised flux", "current multiplier")
    normalised = np.linspace(0.0, 1.0, 300)
    for index, (colour, (name, profile_width)) in enumerate(
        zip(colours, PROFILE_SPECS, strict=True)
    ):
        if profile_width is None:
            multiplier = np.ones_like(normalised)
        else:
            multiplier = 1.0 + PEDESTAL_AMPLITUDE / (
                1.0 + np.exp(-(normalised - PEDESTAL_CENTRE) / profile_width)
            )
        points = " ".join(
            f"{x + value * panel_width:.2f},{y + panel_height - (scale - 1.0) / 1.5 * panel_height:.2f}"
            for value, scale in zip(normalised, multiplier, strict=True)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2.2"/>'
        )
        lines.append(
            f'<text x="{x + 14}" y="{y + 20 + 18 * index}" fill="{colour}" font-size="12">{name.replace("_", " ")}</text>'
        )

    for (x, y), quantity in zip(panels[1:], ("psi", "b_r", "b_z"), strict=True):
        draw_axes(
            x,
            y,
            quantity.replace("_", " "),
            "mesh pitch [m]",
            "sup error / reference span",
        )
        series = []
        for colour, (_profile, rows) in zip(
            colours, findings["cases"].items(), strict=True
        ):
            pitch = np.asarray([row["pitch_m"] for row in rows])
            linear = np.asarray(
                [
                    row["linear_representation"][quantity]["sup_fraction_of_span"]
                    for row in rows
                ]
            )
            centroid = np.asarray(
                [
                    row["centroid_production"][quantity]["sup_fraction_of_span"]
                    for row in rows
                ]
            )
            series.extend(
                ((pitch, linear, colour, False), (pitch, centroid, colour, True))
            )
        all_x = np.log10(np.concatenate([item[0] for item in series]))
        all_y = np.log10(np.concatenate([item[1] for item in series]))
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
        y_pad = max(0.08 * (y_max - y_min), 0.1)
        y_min, y_max = y_min - y_pad, y_max + y_pad
        for pitch, error, colour, dashed in series:
            px = x + (np.log10(pitch) - x_min) / (x_max - x_min) * panel_width
            py = (
                y
                + panel_height
                - (np.log10(error) - y_min) / (y_max - y_min) * panel_height
            )
            points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(px, py, strict=True))
            dash = ' stroke-dasharray="7 5" opacity="0.65"' if dashed else ""
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"{dash}/>'
            )
            for a, b in zip(px, py, strict=True):
                lines.append(
                    f'<circle cx="{a:.2f}" cy="{b:.2f}" r="3" fill="{colour}"/>'
                )
        lines.append(
            f'<text x="{x + 8}" y="{y + 17}" font-size="11" fill="#555">solid: linear moments; dashed: centroid production</text>'
        )
    lines.append("</svg>")
    (output / "steep_edge_moment_convergence.svg").write_text("\n".join(lines) + "\n")


def run(output: Path) -> dict[str, Any]:
    configure_dtypes()
    case = reference_cases()["moderate-rotation-conventional"].static_limit()
    targets = _targets(case)
    prior = _integrated_reference(case, targets, REFERENCE_ORDERS[0])
    converged = _integrated_reference(case, targets, REFERENCE_ORDERS[1])
    exact = _exact_solovev(case, targets)
    references = dict(converged)
    references["solovev_exact"] = exact
    reference_convergence = {}
    for name, _width in PROFILE_SPECS:
        comparison = exact if name == "solovev_exact" else converged[name]
        reference_convergence[name] = _metrics(
            prior[name] + (exact - converged[name] if name == "solovev_exact" else 0.0),
            comparison,
        )

    cases: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in PROFILE_SPECS}
    timing_inputs = None
    for pitch in PITCHES:
        mesh, cells, packed = _mesh(case, pitch)
        shared, cell_node = packed
        atomic = AtomicCellMesh.from_cells(cells, centroids=mesh.coordinate)
        clipped = atomic.clip(atomic.sample(case.flux))
        if not clipped.contour_closed:
            raise AssertionError(
                "analytic contour did not close inside the source mesh"
            )
        area_residual = (
            abs(clipped.patch_area_sum - clipped.contour_area) / clipped.contour_area
        )
        if area_residual > AREA_TOLERANCE:
            raise AssertionError(
                f"patch-area conservation failed at pitch {pitch}: {area_residual}"
            )
        block = _blocks(targets, mesh, cells)
        centroid_inside = np.asarray(
            case.contains(mesh.coordinate[:, 0], mesh.coordinate[:, 1])
        )
        for name, width in PROFILE_SPECS:
            density = _density_function(case, width)
            vectors, centroid_density, gradient = _vectors(
                mesh, clipped, shared, cell_node, density, pitch
            )
            linear_plasma = _contract(block, vectors)
            centroid_vector = centroid_density * mesh.cell_area * centroid_inside
            centroid_plasma = np.asarray(
                [block[q, 0] @ centroid_vector for q in range(3)]
            )
            complement = exact - converged[name] if name == "solovev_exact" else 0.0
            linear = linear_plasma + complement
            centroid = centroid_plasma + complement
            cases[name].append(
                {
                    "pitch_m": pitch,
                    "cell_count": mesh.node_count,
                    "boundary_cell_count": int(np.count_nonzero(clipped.boundary)),
                    "any_intersection_centroid_outside_count": int(
                        np.count_nonzero(clipped.included & ~centroid_inside)
                    ),
                    "patch_area_relative_residual": float(area_residual),
                    "linear_representation": _metrics(linear, references[name]),
                    "centroid_production": _metrics(centroid, references[name]),
                }
            )
            if pitch == PITCHES[1] and name == "pedestal_steep":
                timing_inputs = (case, mesh, cells, centroid_density, gradient)

    fitted = {}
    improvement = {}
    for name, rows in cases.items():
        fitted[name] = {
            "linear_representation": _fit_orders(rows, "linear_representation"),
            "centroid_production": _fit_orders(rows, "centroid_production"),
        }
        improvement[name] = {}
        coarse = rows[0]
        for quantity in ("psi", "b_r", "b_z"):
            improvement[name][quantity] = {}
            for norm in ("sup_fraction_of_span", "rms_fraction_of_span"):
                improvement[name][quantity][norm] = float(
                    coarse["centroid_production"][quantity][norm]
                    / coarse["linear_representation"][quantity][norm]
                )
    if timing_inputs is None:
        raise AssertionError("timing mesh was not selected")
    timing = _iteration_timing(*timing_inputs)
    area_max = max(
        row["patch_area_relative_residual"] for rows in cases.values() for row in rows
    )
    factors = [
        improvement[name][quantity][norm]
        for name, _ in PROFILE_SPECS
        for quantity in ("psi", "b_r", "b_z")
        for norm in ("sup_fraction_of_span", "rms_fraction_of_span")
    ]
    findings = {
        "schema": "nova-steep-edge-moment-demonstration",
        "reference": {
            "analytic_family": "static Solov'ev member of the rotating-equilibrium reference family",
            "solovev_fields": "closed-form total flux and analytic derivatives, with the converged plasma integral resolving the homogeneous complement",
            "pedestal_fields": "64x128 Gauss-Legendre integration in exact analytic flux coordinates",
            "nested_orders": [list(value) for value in REFERENCE_ORDERS],
            "convergence": reference_convergence,
            "target_count": len(targets),
            "target_surface": "an exterior oval at 1.18 radial and 1.30 vertical LCFS scale",
        },
        "representation": "nine exact fixed polygon blocks with degree-three interior moments and conservative any-intersection clipped boundary moments",
        "profiles": {
            name: {"pedestal_width_psi_n": width} for name, width in PROFILE_SPECS
        },
        "cases": cases,
        "fitted_orders": fitted,
        "coarsest_pitch_improvement_factors": improvement,
        "live_checks": {
            "area_relative_tolerance": AREA_TOLERANCE,
            "maximum_patch_area_relative_residual": area_max,
            "all_cases_conserve_area": area_max <= AREA_TOLERANCE,
            "all_cases_use_any_intersection": all(
                row["any_intersection_centroid_outside_count"] > 0
                for rows in cases.values()
                for row in rows
            ),
            "minimum_coarsest_improvement_factor": min(factors),
            "maximum_coarsest_improvement_factor": max(factors),
        },
        "moving_separatrix_timing": timing,
    }
    output.mkdir(parents=True, exist_ok=True)
    _plot(findings, output)
    (output / "steep_edge_moment_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True) + "\n"
    )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    findings = run(arguments.output)
    print(json.dumps(findings["live_checks"], indent=2, sort_keys=True))
    print(json.dumps(findings["moving_separatrix_timing"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
