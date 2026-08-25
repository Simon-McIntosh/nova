# Diverted-root receipt composition delta

## Verdict

`qualify(write=False)` completed on the banked diverted root with the CPU backend, x64 enabled, and a warm carrier-cache hit. Of the 26 scalar leaves under `composition`, 16 are exactly identical and 10 differ: two SHA-256 fields and eight numeric leaves. The eight numeric differences are four distinct quantities repeated for the pinned and unpinned normalization anchors.

The drift is **not confined to digests or provenance**. It reaches roundoff-scale physical values:

- axis radial coordinate: `+2.220446049250313e-16 m`;
- axis vertical coordinate: `+4.163336342344337e-17 m`;
- axis flux: `-8.673617379884035e-19 Wb`;
- boundary flux: `+3.231430693775156e-19 Wb`.

Pinned and unpinned anchors move together, so their axis and boundary distances, flux differences, and domain-label difference count all remain exactly zero. The composition closure remains `4.440892098500626e-16 Wb`; the external and source maximum absolute fluxes and both source-gradient scalars are exactly unchanged.

The root result is unaffected. The other five top-level receipt items—`schema`, `schema_version`, `state`, `map`, and `evidence`—are exactly equal to the bank. In particular, the banked state still maps requested/achieved class `1/1`, is topology-consistent, converges in one application, and has relative residual `1.5840538799920246e-16` and absolute/terminal difference `4.440892098500626e-16 Wb`.

## Complete leaf comparison

Every numeric leaf is shown with signed delta `measured - banked` and absolute delta. Digest deltas are not numeric; their equality result is shown explicitly.

| Composition field | Measured | Banked | Delta | Absolute delta | Equal |
|---|---:|---:|---:|---:|:---:|
| `closure_absolute_residual_wb` | `4.440892098500626e-16` | `4.440892098500626e-16` | `0.0` | `0.0` | yes |
| `external_field.maximum_absolute_flux_wb` | `2.8331924006294433` | `2.8331924006294433` | `0.0` | `0.0` | yes |
| `external_field.reconstruction_difference_wb` | `0.0` | `0.0` | `0.0` | `0.0` | yes |
| `external_field.sha256` | `d6941b63cd30c1a60b31cd18bb3f473e27c500295fa0155251583ae6c23c69e6` | `b1a26f6828854302e6a62bb18938e3e5f908630dff06a50783353c2e2df47463` | n/a | n/a | **no** |
| `normalization_anchor.axis_distance_m` | `0.0` | `0.0` | `0.0` | `0.0` | yes |
| `normalization_anchor.axis_flux_difference_wb` | `0.0` | `0.0` | `0.0` | `0.0` | yes |
| `normalization_anchor.boundary_distance_m` | `0.0` | `0.0` | `0.0` | `0.0` | yes |
| `normalization_anchor.boundary_flux_difference_wb` | `0.0` | `0.0` | `0.0` | `0.0` | yes |
| `normalization_anchor.domain_label_difference_count` | `0` | `0` | `0` | `0` | yes |
| `normalization_anchor.pinned_axis_flux_wb` | `0.006820219358913838` | `0.0068202193589138385` | `-8.673617379884035e-19` | `8.673617379884035e-19` | **no** |
| `normalization_anchor.pinned_axis_m[0]` | `1.7627354924305096` | `1.7627354924305094` | `2.220446049250313e-16` | `2.220446049250313e-16` | **no** |
| `normalization_anchor.pinned_axis_m[1]` | `-0.06199979001032992` | `-0.06199979001032996` | `4.163336342344337e-17` | `4.163336342344337e-17` | **no** |
| `normalization_anchor.pinned_boundary_flux_wb` | `-1.484486541947421e-06` | `-1.4844865419477442e-06` | `3.231430693775156e-19` | `3.231430693775156e-19` | **no** |
| `normalization_anchor.pinned_boundary_m[0]` | `1.2043590455560238` | `1.2043590455560238` | `0.0` | `0.0` | yes |
| `normalization_anchor.pinned_boundary_m[1]` | `-0.43060062587647197` | `-0.43060062587647197` | `0.0` | `0.0` | yes |
| `normalization_anchor.unpinned_axis_flux_wb` | `0.006820219358913838` | `0.0068202193589138385` | `-8.673617379884035e-19` | `8.673617379884035e-19` | **no** |
| `normalization_anchor.unpinned_axis_m[0]` | `1.7627354924305096` | `1.7627354924305094` | `2.220446049250313e-16` | `2.220446049250313e-16` | **no** |
| `normalization_anchor.unpinned_axis_m[1]` | `-0.06199979001032992` | `-0.06199979001032996` | `4.163336342344337e-17` | `4.163336342344337e-17` | **no** |
| `normalization_anchor.unpinned_boundary_flux_wb` | `-1.484486541947421e-06` | `-1.4844865419477442e-06` | `3.231430693775156e-19` | `3.231430693775156e-19` | **no** |
| `normalization_anchor.unpinned_boundary_m[0]` | `1.2043590455560238` | `1.2043590455560238` | `0.0` | `0.0` | yes |
| `normalization_anchor.unpinned_boundary_m[1]` | `-0.43060062587647197` | `-0.43060062587647197` | `0.0` | `0.0` | yes |
| `source_forcing.ff_prime_t2_m2_per_wb` | `-0.08105694691387022` | `-0.08105694691387022` | `0.0` | `0.0` | yes |
| `source_forcing.maximum_absolute_flux_wb` | `0.14172801675859822` | `0.14172801675859822` | `0.0` | `0.0` | yes |
| `source_forcing.p_prime_pa_per_wb` | `-16125.767218728875` | `-16125.767218728875` | `0.0` | `0.0` | yes |
| `source_forcing.repeat_difference_wb` | `0.0` | `0.0` | `0.0` | `0.0` | yes |
| `source_forcing.sha256` | `7e3a16ff57c5ba6d85e3e3a6bf8de0d17e097529f2200c4f9553e87173c61f7c` | `8a6bd7a20414c141ae7c8d2d7b0fe51451bfb61c9cc32fc2321b9b642d08bb89` | n/a | n/a | **no** |

## What the digests cover

Both digests are computed by `sha256(np.ascontiguousarray(values).tobytes())` in the current qualifier and in the commit that banked this receipt. Therefore they cover only the contiguous raw bytes of the evaluated numeric arrays:

- `external_field.sha256` covers `operator.external()`;
- `source_forcing.sha256` covers the repeated `operator.internal(state, requested_class)` source image.

They do **not** cover a cache file, file timestamps, paths, serialization headers, or other write metadata. Consequently, the changed digests prove that both evaluated arrays are not bitwise identical to the banked arrays. A metadata-only digest explanation is excluded for this receipt. The bank retains the digests and scalar summaries, not the old arrays themselves, so a pointwise array delta cannot be recovered from the surviving bank without regenerating historical data; no such regeneration was performed.

The differing array identities do not alter the receipt's stored physical summaries: external maximum absolute flux is exactly unchanged at `2.8331924006294433 Wb`, source maximum absolute flux is exactly unchanged at `0.14172801675859822 Wb`, external reconstruction difference and source repeat difference remain exactly `0.0 Wb`, and composition closure remains `4.440892098500626e-16 Wb`.

## Reproduction

The measurement used `JAX_PLATFORMS=cpu`, the repository's shared Nova environment, a writable per-run cache, and `qualify(write=False)`. Full stdout, all 26 leaf comparisons, both complete composition objects, and `EXIT_MARKER=0` are preserved at:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T102051495285-receipt-composition-delta/qualify-composition.log`

No source, test, state bank, JSON receipt, or other banked artifact was written or regenerated.
