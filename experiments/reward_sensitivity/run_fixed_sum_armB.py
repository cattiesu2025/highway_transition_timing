"""Arm B extension: 35 new tasks; existing F3/R2 and F5/BAL are reused."""
from pathlib import Path
import run_fixed_sum as fixed

BASE_SEEDS = {'B': 4100}
MANIFEST = Path(__file__).with_name('jobs_fixed_sum_armB.json')


def main(argv=None):
    return fixed.main(argv, task_list=fixed.jobs(BASE_SEEDS), manifest=MANIFEST)


if __name__ == '__main__':
    raise SystemExit(main())
