"""Development follow-up: two equal-total-weight FD/SP comparisons in A/C."""
from __future__ import annotations
import run as sweep

VARIANTS = {'R7': ('FD', .25, 1.), 'R8': ('SP', 1., .45)}
BASE_SEEDS = {'A': 3100, 'C': 4200}


def jobs():
    return [dict(index=i, arm=arm, variant=variant, agent=values[0],
                 speed_weight=values[1], front_distance_weight=values[2], seed=seed)
            for i, (arm, variant, values, seed) in enumerate(
                (arm, variant, values, base + offset)
                for arm, base in BASE_SEEDS.items()
                for variant, values in VARIANTS.items() for offset in range(5))]


if __name__ == '__main__':
    raise SystemExit(sweep.main(task_list=jobs()))
