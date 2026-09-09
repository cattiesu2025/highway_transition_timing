# Held-Out V2 Protocol

> Status: **frozen before inference on 2026-09-09 AEST**. Grid checksums are
> recorded below. No v2 rollout may begin while the repository tests fail.

## Scope

Held-out v2 evaluates the already selected policies from the current runs:

| Arm | Run prefix | Seeds | Endpoint | Grid |
| --- | --- | --- | --- | --- |
| A | `single_lane_slow_front_fixed_100k_seed` | 3100-3119 | persistent slowdown onset | `arm_ab_grid.csv` |
| B | `multilane_twolane_open_fixed_100k_seed` | 4100-4119 | physical lane-index change | `arm_ab_grid.csv` |
| C | `multilane_twolane_occupied_fixed_100k_seed` | 4200-4219 | physical lane-index change | `arm_c_grid.csv` |

There is no training, checkpoint selection, model replacement, or parameter
tuning in this stage. Each selected model is evaluated deterministically on the
same 36 exposures as the other reward conditions in its arm.

## Sealed Grids

`arm_ab_grid.csv` is the full factorial:

- ego speed: 25, 27, 29 m/s;
- front centre distance: 142.5, 157.5, 172.5, 187.5 m;
- front speed: 11, 15, 19 m/s.

`arm_c_grid.csv` is the full factorial:

- fixed ego speed: 28 m/s;
- front centre distance: 142.5, 157.5, 172.5, 187.5 m;
- front speed: 11, 15, 19 m/s;
- target-lane merge gap: 25, 55, 85 m.

The A/B grid uses exposure IDs `H2AB0000-H2AB0035`; the C grid uses
`H2C0000-H2C0035`. Both use fixed exposure seeds `9100000-9100035`, independent
of training seed. The A/B checksum is
`094a840d57782fe63eba6604c8ac36cc45e5b1d4f5aa72d05a2cba4fe5a18caf`;
the C checksum is
`065af8ea3de7bfa8f56695e81675b553ea3047158e9f694c746c7fe8b30cfdf9`.

## Confirmatory Decisions

For every training seed, first compute each policy's median valid onset latency
and each exposure-matched pairwise latency gap. Across the 20 seeds, report the
median, percentile-bootstrap 95% interval, and positive/negative/tied seed
counts. Whole seeds, not exposure rows, are resampled.

The two primary predictions are:

1. Arm A: `SP - FD > 0` seconds.
2. Arm C: `SP - FD < 0` seconds.

The reversal claim is confirmed only if both intervals lie wholly in their
predeclared directions. Supporting predictions are Arm A `BAL - FD > 0` and
Arm C `SP - BAL < 0`, judged by the same rule. Arm B has no directional or
equivalence success criterion; all three contrasts are descriptive controls.

No-onset episodes remain censored and are not assigned an artificial latency.
The report must show the observed/total denominator and collisions for every
agent and variant. A seed contrast with fewer than 24 of 36 jointly observed
pairs is non-estimable. A confirmatory contrast without all 20 estimable seed
summaries is also non-estimable rather than forced into a timing claim.

## Counterfactuals And Integrity

All arms run `original`, `no-front`, `matched-speed-front`, and `far-front`.
Arm C additionally runs `open-target-lane`. These are supporting specificity
and mechanism checks, not replacements for the primary held-out ordering.

Before aggregation, every run must contain all three selected model archives,
the training metadata, a sealed-grid manifest with the expected SHA-256, and
36 unique exposure IDs in every requested variant. Any mismatch stops the
aggregation. Existing v1 files and outputs remain untouched and cannot be
combined with v2.
