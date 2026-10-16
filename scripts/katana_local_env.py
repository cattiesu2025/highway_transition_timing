"""Run one job in an unchanged node-local venv, serializing this user's node jobs.

Invoke with the module Python and -S so staging needs no third-party imports.
The lock stays held until the child exits and its temporary environment is removed.
"""
from __future__ import annotations

import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import time


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
    print(f'Node space: available={available / 1024**3:.2f} GiB; '
          f'required including reserve={required_bytes / 1024**3:.2f} GiB; '
          f'free inodes={fs.f_favail}; required inodes={required_inodes}', flush=True)
    if available < required_bytes or (fs.f_files and fs.f_favail < required_inodes):
        raise RuntimeError('Insufficient node-local space/inodes; refusing extraction before training')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--local-root', type=Path, default=Path('/tmp'))
    parser.add_argument('--lock-timeout', type=float, default=14400)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required after --')
    if args.lock_timeout < 0:
        parser.error('--lock-timeout must be nonnegative')
    bundle = args.bundle.resolve()
    archive = bundle / 'venv.tar'
    if not archive.is_file() or not (bundle / 'venv.tar.sha256').is_file():
        raise FileNotFoundError('Both venv.tar and venv.tar.sha256 are required')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminate)
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
        check_space(args.local_root, needed, inodes)
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
