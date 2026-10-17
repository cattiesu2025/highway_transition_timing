"""Run jobs in an unchanged node-local venv.

Invoke with module Python and -S (stdlib only). Default legacy mode holds the
node lock for the child lifetime. --shared-env locks preparation only and
retains a checksum-keyed cache so children can run concurrently.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import re
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import tarfile
import tempfile
import time


class InsufficientSpace(RuntimeError):
    pass


def log_space(message):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] node={socket.gethostname()} {message}", flush=True)


class Terminated(RuntimeError):
    pass


def terminate(signum, frame):
    raise Terminated(f'Received signal {signum}')


def run_bounded(command, *, timeout, **kwargs):
    """Stop the whole child process group on timeout or job termination."""
    child = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        code = child.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def acquire_lock(root, wait_seconds):
    # Never unlink the lock file: waiters must always refer to the same inode.
    directory = root / f'highway-env-lock-{os.getuid()}'
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f'Unsafe lock directory: {directory}')
    fd = os.open(directory / 'lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    handle = os.fdopen(fd, 'w')
    deadline = time.monotonic() + wait_seconds
    next_message = 0
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print('Node environment lock acquired.', flush=True)
                return handle
            except BlockingIOError:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError('Timed out waiting for this user\'s node environment lock')
                if now >= next_message:
                    print('Waiting for another local-environment job on this node; no environment copied yet.', flush=True)
                    next_message = now + 60
                time.sleep(min(1, deadline - now))
    except BaseException:
        handle.close()
        raise


def required_space(archive):
    allocated, entries = 0, 0
    with tarfile.open(archive, 'r:') as bundle:
        for member in bundle:
            entries += 1
            allocated += max(4096, math.ceil(member.size / 4096) * 4096)
    # Cache growth, filesystem metadata, and Python bytecode headroom.
    return allocated + 512 * 1024**2, entries + 4096


def check_space(root, required_bytes, required_inodes):
    available = shutil.disk_usage(root).free
    fs = os.statvfs(root)
    log_space(f'Node space: available={available / 1024**3:.2f} GiB; '
          f'required including reserve={required_bytes / 1024**3:.2f} GiB; '
          f'free inodes={fs.f_favail}; required inodes={required_inodes}')
    if available < required_bytes or (fs.f_files and fs.f_favail < required_inodes):
        raise InsufficientSpace('Insufficient node-local space/inodes; refusing extraction before training')


def wait_for_space(root, required_bytes, required_inodes, timeout=1800, interval=60):
    """Keep the node lock; wait without extracting or creating a training folder."""
    if not math.isfinite(timeout) or timeout < 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError('Space timeout must be finite/nonnegative; interval finite/positive')
    start = time.monotonic()
    deadline = start + timeout
    log_space(f'Space wait started: path={root}; limit={timeout:.0f}s; interval={interval:.0f}s')
    while True:
        try:
            check_space(root, required_bytes, required_inodes)
        except InsufficientSpace:
            now = time.monotonic()
            remaining = max(0, deadline - now)
            if not remaining:
                log_space(f'Space wait timed out: elapsed={now-start:.1f}s; '
                          'no extraction or training started; releasing node lock on exit')
                raise InsufficientSpace('Insufficient node-local space/inodes after bounded wait')
            delay = min(interval, remaining)
            log_space(f'Space unavailable: elapsed={now-start:.1f}s; remaining={remaining:.1f}s; '
                      f'next check in {delay:.1f}s')
            time.sleep(delay)
        else:
            log_space(f'Space ready: elapsed={time.monotonic()-start:.1f}s; proceeding to extraction')
            return


def private_directory(path):
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f'Unsafe cache directory: {path}')
    return path


def prepare_shared_environment(args, bundle, archive):
    """Publish one verified immutable snapshot per node/user/hash; never evict it here.

    Retaining the cache avoids deleting an environment while another process is
    importing from it. Only preparation holds the lock; training never does.
    """
    checksum = (bundle / 'venv.tar.sha256').read_text().strip()
    match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *]venv\.tar', checksum)
    if not match:
        raise ValueError('Expected one SHA256 entry for venv.tar')
    digest = match.group(1).lower()
    cache_root = private_directory(args.local_root / f'highway-shared-env-{os.getuid()}')
    cache = cache_root / digest
    with acquire_lock(args.local_root, args.lock_timeout):
        if cache.exists():
            private_directory(cache)
            marker = json.loads((cache / 'ready.json').read_text())
            if marker != {'sha256': digest, 'version': 1} or not (cache / 'venv' / 'pyvenv.cfg').is_file():
                raise RuntimeError(f'Invalid shared cache; refusing reuse: {cache}')
            log_space(f'Shared environment cache hit: {cache}; preparation lock released before child')
            return cache / 'venv'
        log_space(f'Shared environment cache miss: {digest}; staging once for this node')
        run_bounded(['sha256sum', '-c', 'venv.tar.sha256'], cwd=bundle, timeout=300)
        signal.signal(signal.SIGALRM, terminate)
        signal.alarm(300)
        try:
            needed, inodes = required_space(archive)
        finally:
            signal.alarm(0)
        wait_for_space(args.local_root, needed, inodes, args.space_timeout, args.space_check_interval)
        with tempfile.TemporaryDirectory(prefix='.staging-', dir=cache_root) as work:
            staging = Path(work)
            (staging / 'venv').mkdir()
            run_bounded(['tar', '-xf', str(archive), '-C', str(staging / 'venv')], timeout=300)
            if not (staging / 'venv' / 'pyvenv.cfg').is_file():
                raise RuntimeError('Extracted archive is not a Python venv')
            (staging / 'ready.json').write_text(json.dumps({'sha256': digest, 'version': 1}))
            # Atomic publication: waiting jobs never see a partially extracted venv.
            staging.rename(cache)
        log_space(f'Shared environment ready: {cache}; preparation lock released before child')
    return cache / 'venv'


def run_shared(args, bundle, archive, command):
    venv = prepare_shared_environment(args, bundle, archive)
    # Cache storage is shared; all writable runtime/fontconfig caches remain per-job.
    with tempfile.TemporaryDirectory(prefix='highway-shared-job.', dir=args.local_root) as work:
        env = dict(os.environ, VIRTUAL_ENV=str(venv), EXPECTED_LOCAL_VENV=str(venv),
                   TMPDIR=work, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   PATH=f'{venv}/bin:' + os.environ['PATH'])
        env.pop('PYTHONHOME', None)
        log_space(f'Starting concurrent child with shared environment; private TMPDIR={work}')
        try:
            run_bounded(command, env=env, timeout=None)
        finally:
            log_space('Child finished; removing private job cache only, retaining shared environment')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-env', action='store_true', help='Reuse a verified node cache; unlock before training')
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--local-root', type=Path, default=Path('/tmp'))
    parser.add_argument('--lock-timeout', type=float, default=10800)
    parser.add_argument('--space-timeout', type=float, default=1800)
    parser.add_argument('--space-check-interval', type=float, default=60)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required after --')
    for name in ('lock_timeout', 'space_timeout', 'space_check_interval'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == 'space_check_interval' and value == 0):
            parser.error(f'Invalid --{name.replace("_", "-")}: {value}')
    bundle = args.bundle.resolve()
    archive = bundle / 'venv.tar'
    if not archive.is_file() or not (bundle / 'venv.tar.sha256').is_file():
        raise FileNotFoundError('Both venv.tar and venv.tar.sha256 are required')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminate)
    if args.shared_env:
        run_shared(args, bundle, archive, command)
        return 0
    with acquire_lock(args.local_root, args.lock_timeout):
        # Read the shared archive directly, never copy its 1.2 GB into /tmp.
        run_bounded(['sha256sum', '-c', 'venv.tar.sha256'], cwd=bundle, timeout=300)
        # Bound metadata scanning too, since the archive lives on shared storage.
        signal.signal(signal.SIGALRM, terminate)
        signal.alarm(300)
        try:
            needed, inodes = required_space(archive)
        finally:
            signal.alarm(0)
        wait_for_space(args.local_root, needed, inodes, args.space_timeout, args.space_check_interval)
        with tempfile.TemporaryDirectory(prefix='highway-env-job.', dir=args.local_root) as work:
            local = Path(work)
            venv = local / 'venv'
            venv.mkdir()
            (local / 'tmp').mkdir()
            print(f'Extracting unchanged environment into {venv}', flush=True)
            run_bounded(['tar', '-xf', str(archive), '-C', str(venv)], timeout=300)
            env = dict(os.environ, VIRTUAL_ENV=str(venv), EXPECTED_LOCAL_VENV=str(venv),
                       TMPDIR=str(local / 'tmp'), PYTHONNOUSERSITE='1',
                       PATH=f'{venv}/bin:' + os.environ['PATH'])
            env.pop('PYTHONHOME', None)
            run_bounded(command, env=env, timeout=None)
        print('Local environment removed; releasing node lock.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
