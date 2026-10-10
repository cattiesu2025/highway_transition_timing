"""Prepare job-local fontconfig and bound dependency startup before training.

Called under GNU timeout by the PBS entry points. No simulator is run and no
training output directory is created. Uses only Matplotlib's bundled fonts
for fontconfig discovery; does not edit global or user font configuration.
"""
from __future__ import annotations
import faulthandler
import importlib
import importlib.util
import os
import sys
from pathlib import Path
import time
import xml.etree.ElementTree as ET


def prepare_fontconfig():
    spec = importlib.util.find_spec('matplotlib')
    if spec is None or spec.origin is None:
        raise RuntimeError('Matplotlib is not installed in the training environment')
    fonts = Path(spec.origin).parent/'mpl-data'/'fonts'/'ttf'
    if not fonts.is_dir():
        raise FileNotFoundError(f'Matplotlib bundled fonts missing: {fonts}')
    config = Path(os.environ['FONTCONFIG_FILE'])
    cache = Path(os.environ['XDG_CACHE_HOME'])/'fontconfig'
    config.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    root = ET.Element('fontconfig')
    ET.SubElement(root, 'dir').text = str(fonts)
    ET.SubElement(root, 'cachedir').text = str(cache)
    ET.ElementTree(root).write(config, encoding='utf-8', xml_declaration=True)
    print(f'Fontconfig: {config}; font directory: {fonts}', flush=True)


def enable_open_diagnostics():
    """Log Python open attempts, without ptrace or logging file contents.

    An event is emitted before the operation: it is not a read-completion
    record, and the last path alone cannot prove which operation blocked.
    """
    if os.environ.get('HEADLESS_TRACE_OPENS') != '1':
        return

    def log_open(event, args):
        if event == 'open':
            print(f'PYTHON_OPEN t={time.monotonic():.3f} path={args[0]!r} '
                  f'mode={args[1]!r}', file=sys.stderr, flush=True)

    sys.addaudithook(log_open)
    print('Python open diagnostics enabled (no file contents).', flush=True)


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        enable_open_diagnostics()
        prepare_fontconfig()
        for name in ('matplotlib.pyplot', 'stable_baselines3', 'highway_env'):
            start = time.monotonic()
            print(f'Preflight importing {name}', flush=True)
            importlib.import_module(name)
            print(f'Preflight imported {name} in {time.monotonic()-start:.1f}s', flush=True)
        print('Preflight passed.', flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
