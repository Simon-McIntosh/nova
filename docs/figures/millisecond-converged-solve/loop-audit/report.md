# Loop-structure audit of the whole-cell certificate solve

Compiled (not executed) on the CPU backend (`JAX_PLATFORMS=cpu`) for the weak row at 300 cells.  Each budget row doubles that one budget against the certificate baseline (newton_steps=10, gmres_iterations=30, warmup=0, active_set_steps=16, capacity 30, 300 nodes).

| budget | baseline | doubled | base instrs | doubled instrs | ratio | verdict |
|---|---|---|---|---|---|---|
| newton_steps | 10 | 20 | 657383 | 657383 | 1.000 | scanned_or_fixed |
| gmres_iterations | 30 | 60 | 657383 | 657463 | 1.000 | scanned_or_fixed |
| warmup | 0 | 0 | 657383 | 657383 | 1.000 | degenerate_zero (pinned by the certificate) |
| active_set_steps | 16 | 32 | 657383 | 657419 | 1.000 | scanned_or_fixed |
| backtracking_factors | 6 | 12 | 657383 | 657383 | 1.000 | scanned_or_fixed |
| model_rebuild_damping_trips | 6 | 12 | 657383 | 657383 | 1.000 | scanned_or_fixed |
| topology_table_capacity | 30 | 60 | 657383 | 659116 | 1.003 | scanned_or_fixed |
| profile_node_count | 300 (342 nodes) | None | None | None | -- | not measurable: no doubled cell count was buildable |

## Python loop census inside traced functions

Every Python `for`/`while` and comprehension in the four files, with the bound each iterates.  A `for` over `range(newton_steps)` or `range(active_set_steps)` is a budget-iteration unroll at trace time; a `for` over static structure (pairs, kernels, stencils) is fixed. 
### nova/equilibrium/fixed_point.py

| line | construct | function | bound iterable |
|---|---|---|---|
| 223 | DictComp | load_fixed_point_checkpoint | `checkpoint.files` |
| 4057 | GeneratorExp | newton_krylov | `row_jvp_observers` |

### nova/equilibrium/reduced_newton.py

| line | construct | function | bound iterable |
|---|---|---|---|
| 1000 | For | _first_accept_grades | `enumerate(_BACKTRACKING_FACTORS)` |
| 1127 | For | _plain_newton_trip | `range(newton_steps)` |
| 1301 | For | _drive_trips | `range(active_set_steps)` |
| 1924 | While | _remember_compiled_program | `condition` |
| 1959 | For | _bind_dynamic_arguments | `kernels.items()` |
| 2122 | For | _row_augmentation | `pairs` |
| 2275 | For | linearized_constraint_response_matrix | `pairs` |
| 2329 | For | derive_reduced_constraint_pairs | `zip(pairs, row_slices, strict=True)` |
| 3031 | For | solve_constrained_reduced_newton_compiled | `zip(pairs, records, strict=True)` |
| 400 | DictComp | _scatter | `zip(coordinates.leaves, blocks, strict=True)` |
| 1166 | For | _plain_newton_trip | `range(2)` |
| 1943 | DictComp | _bind_rows | `kernels.items()` |
| 2000 | For | flux_delta | `zip(pairs, self.row_slices, strict=True)` |
| 392 | ListComp | _gather | `coordinates.leaves` |
| 2487 | GeneratorExp | solve_constrained_reduced_newton | `pairs` |
| 2662 | For | solve_constrained_reduced_newton | `zip(pairs, records, strict=True)` |
| 2728 | ListComp | _compiled_output_fields | `host[10][:iterations]` |
| 2729 | ListComp | _compiled_output_fields | `host[11][:iterations]` |
| 2730 | ListComp | _compiled_output_fields | `host[12][:iterations]` |
| 2731 | ListComp | _compiled_output_fields | `host[13][:iterations]` |
| 2732 | ListComp | _compiled_output_fields | `host[14][:iterations]` |
| 2733 | ListComp | _compiled_output_fields | `host[15][:iterations]` |
| 3024 | GeneratorExp | solve_constrained_reduced_newton_compiled | `records` |
| 405 | GeneratorExp | _scatter | `CellCurrentMoments._fields` |
| 2323 | GeneratorExp | derive_reduced_constraint_pairs | `pairs` |
| 2658 | GeneratorExp | solve_constrained_reduced_newton | `records` |
| 2754 | GeneratorExp | _compiled_program | `augmentation.pairs` |
| 3043 | GeneratorExp | solve_constrained_reduced_newton_compiled | `records` |
| 2014 | GeneratorExp | rows | `zip(pairs, self.row_slices, strict=True)` |
| 2557 | GeneratorExp | solve_constrained_reduced_newton | `pairs` |
| 2676 | GeneratorExp | solve_constrained_reduced_newton | `records` |
| 2860 | GeneratorExp | _compiled_result | `augmentation.pairs` |
| 2611 | GeneratorExp | solve_constrained_reduced_newton | `pairs` |

### nova/equilibrium/forward.py

| line | construct | function | bound iterable |
|---|---|---|---|
| 1135 | For | constraint_residual | `pins.moments` |
| 1627 | For | _solve_host | `range(budget)` |
| 2227 | While | _solve_eliminated_request | `condition` |
| 213 | GeneratorExp | _lattice_cells | `lattice.coordinate` |
| 1119 | For | constraint_residual | `pins.isoflux` |
| 2002 | For | _solve_augmented_constraints | `zip(pairs, records, strict=True)` |
| 2176 | For | solve_target | `enumerate(request.constraint_pairs)` |
| 2215 | For | bracket | `zip(ordered[:-1], ordered[1:], strict=True)` |
| 1917 | GeneratorExp | _solve_augmented_constraints | `pairs` |
| 2130 | GeneratorExp | _solve_request | `enumerate(request.constraint_pairs)` |
| 2432 | GeneratorExp | _request_compilation_cache_key | `request.constraint_pairs` |
| 375 | GeneratorExp | __post_init__ | `axis` |
| 376 | GeneratorExp | __post_init__ | `saddle` |
| 434 | GeneratorExp | __post_init__ | `amplitudes` |
| 1793 | GeneratorExp | _accelerated_history_program | `sorted(options.items())` |
| 1915 | GeneratorExp | _solve_augmented_constraints | `pairs` |
| 2108 | DictComp | _resolve_solve_defaults | `(
                    ("current_pin", current_pin),
                    ("exact_kernels", exact_kernels),
                    ("cached_machine", cached_machine),
                    ("compilation_cache", compilation_cache),
                )` |
| 2222 | GeneratorExp | sampled | `targets` |
| 2230 | GeneratorExp | _solve_eliminated_request | `errors` |
| 2266 | ListComp | _solve_eliminated_request | `bounds` |
| 2286 | GeneratorExp | _solve_eliminated_request | `errors` |
| 2288 | GeneratorExp | _solve_eliminated_request | `bounds` |
| 637 | GeneratorExp | from_lattice | `(
                    plasma_to_grid_r,
                    plasma_to_grid_z,
                    plasma_to_wall_r,
                    plasma_to_wall_z,
                )` |
| 2305 | GeneratorExp | _solve_eliminated_request | `steps` |
| 1736 | GeneratorExp | _solve_accelerated | `_CONSTRAINABLE` |
| 2370 | GeneratorExp | _solve_imposed_request | `terminal_constraints` |
| 2650 | GeneratorExp | _branch_receipt | `equilibrium.constraints` |
| 2277 | GeneratorExp | _solve_eliminated_request | `targets` |

### nova/equilibrium/forward_operator.py

| line | construct | function | bound iterable |
|---|---|---|---|
| 1094 | For | _callable_semantic_identity | `enumerate(closure)` |
| 992 | For | _digest_static_value | `sorted(value, key=str)` |
| 996 | For | _digest_static_value | `enumerate(value)` |
| 1000 | For | _digest_static_value | `sorted(vars(value).items())` |
| 1031 | For | _digest_callable_value | `sorted(value, key=repr)` |
| 1037 | For | _digest_callable_value | `enumerate(value)` |
| 1042 | For | _digest_callable_value | `sorted(value, key=repr)` |
| 1048 | For | _digest_callable_value | `value.co_consts` |
| 1062 | For | _digest_callable_value | `sorted(vars(value).items())` |
| 1283 | For | __post_init__ | `enumerate(self.operators)` |
| 1311 | For | map | `enumerate(self.operators)` |
| 1553 | For | __post_init__ | `self._dynamic_extra_names()` |
| 1727 | For | tree_unflatten | `aux.static_state.items()` |
| 1763 | For | tree_unflatten | `zip(aux.dynamic_extra_names, extra_values, strict=True)` |
| 1777 | For | _build_support_moment_stencils | `(4, 6)` |
| 1796 | For | _build_support_moment_stencils | `stencils` |
| 2056 | For | support_current_moments | `self._support_moment_stencils` |
| 2072 | For | support_flux_coefficients | `self._support_moment_stencils` |
| 2084 | For | sample_flux_field | `self._support_moment_stencils` |
| 1608 | DictComp | _compute_geometry_identity | `self.__dict__.items()` |
| 1649 | DictComp | tree_flatten | `self.__dict__.items()` |
| 576 | ListComp | census_lane | `range(2)` |
| 773 | ListComp | _compatibility_census | `ring_masks` |
| 1558 | GeneratorExp | _dynamic_extra_names | `(
                "declared_axis_flux",
                "declared_boundary_flux",
                "declared_support",
            )` |
| 1773 | ListComp | _build_support_moment_stencils | `self.moment_geometry.polygons` |
| 584 | ListComp | census_lane | `deduplicated` |
| 585 | ListComp | census_lane | `deduplicated` |
| 586 | ListComp | census_lane | `deduplicated` |
| 616 | ListComp | census_lane | `compact_representative_mask` |
| 666 | ListComp | census_lane | `compact_type_agrees` |
| 667 | ListComp | census_lane | `compact_typed_mask` |
| 669 | ListComp | census_lane | `compact_representative_mask` |
| 671 | ListComp | census_lane | `compact_multiplicity` |
| 678 | ListComp | census_lane | `compact_parent_origin` |
| 1125 | GeneratorExp | flatten | `leaves` |
| 1280 | GeneratorExp | __post_init__ | `self.operators` |
| 1286 | GeneratorExp | __post_init__ | `groups.values()` |
| 1408 | ListComp | __post_init__ | `kinds` |
| 1459 | GeneratorExp | __post_init__ | `range(self.grid.node_number)` |
| 1697 | GeneratorExp | tree_flatten | `dynamic_extra_names` |
| 2377 | GeneratorExp | scaled_current_moments | `moments` |
