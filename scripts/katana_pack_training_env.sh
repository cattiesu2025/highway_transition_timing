#!/bin/bash
# Read-only snapshot of the maintained venv; no pip installs or upgrades.
set -euo pipefail
source_venv=${1:-/srv/scratch/${USER}/venvs/highway-transition}
archive_root=${2:-/srv/scratch/${USER}/venv_archives}
test -f "${source_venv}/pyvenv.cfg"
test -x "${source_venv}/bin/python3"
mkdir -p "$archive_root"
bundle_dir=$(mktemp -d "${archive_root}/highway-env.XXXXXX")
echo "Packing unchanged environment: ${source_venv}"
echo "Bundle: ${bundle_dir}"
# A time bound avoids an indefinitely blocked shared-filesystem read.
timeout --signal=TERM --kill-after=10s 900s \
  tar -C "$source_venv" -cf "${bundle_dir}/venv.tar.partial" .
mv "${bundle_dir}/venv.tar.partial" "${bundle_dir}/venv.tar"
(cd "$bundle_dir" && sha256sum venv.tar > venv.tar.sha256)
echo "Snapshot ready. Submit with:"
printf 'qsub -v VENV_BUNDLE=%s,SWEEP_INDEX=2 scripts/katana_reward_local_retry.pbs\n' "$bundle_dir"
