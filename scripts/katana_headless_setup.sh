# Source after activating the training environment. Applies only to this job.
# Keep caches private per job, including when multiple jobs share a node.
job_cache_dir=$(mktemp -d "${TMPDIR:?PBS TMPDIR is required}/reward-headless.XXXXXX")
export MPLCONFIGDIR="${job_cache_dir}/matplotlib"
export XDG_CACHE_HOME="${job_cache_dir}/cache"
export FONTCONFIG_PATH="${job_cache_dir}/fontconfig"
export FONTCONFIG_FILE="${FONTCONFIG_PATH}/fonts.conf"
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYTHONFAULTHANDLER=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FONTCONFIG_PATH"
echo "Headless startup on $(hostname); cache: ${job_cache_dir}"
# If startup hangs, emit Python stacks every minute and fail within ~5 minutes,
# before the runner creates an exclusive experiment output directory.
timeout --signal=TERM --kill-after=10s 300s \
  python3 -u scripts/katana_headless_preflight.py
