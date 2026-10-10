"""Headless startup isolates fontconfig without touching experiment settings."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def test_fontconfig_uses_bundled_fonts_and_private_cache(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('headless_preflight', ROOT/'scripts/katana_headless_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path/'package & fonts'/'matplotlib'
    fonts = package/'mpl-data/fonts/ttf'
    fonts.mkdir(parents=True)
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(package/'__init__.py')))
    config = tmp_path/'job/fontconfig/fonts.conf'
    cache = tmp_path/'job/cache'
    monkeypatch.setenv('FONTCONFIG_FILE', str(config))
    monkeypatch.setenv('XDG_CACHE_HOME', str(cache))
    module.prepare_fontconfig()
    xml = ET.parse(config).getroot()
    assert xml.find('dir').text == str(fonts)
    assert xml.find('cachedir').text == str(cache/'fontconfig')
    assert not xml.findall('include')
    assert (cache/'fontconfig').is_dir()


def test_single_retry_matches_array_command():
    array = (ROOT/'scripts/katana_reward_symmetric.pbs').read_text()
    retry = (ROOT/'scripts/katana_reward_symmetric_retry.pbs').read_text()
    assert '#PBS -J' not in retry
    assert array.split('set -euo pipefail', 1)[1].replace('${PBS_ARRAY_INDEX}', '${SWEEP_INDEX:-2}') == retry.split('set -euo pipefail', 1)[1]
    assert retry.index('source scripts/katana_headless_setup.sh') < retry.index('python3 experiments/')


def test_open_diagnostics_opt_in_and_no_file_contents(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location('headless_diagnostics', ROOT/'scripts/katana_headless_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hooks = []
    monkeypatch.setattr(module.sys, 'addaudithook', hooks.append)
    monkeypatch.delenv('HEADLESS_TRACE_OPENS', raising=False)
    module.enable_open_diagnostics()
    assert hooks == []
    monkeypatch.setenv('HEADLESS_TRACE_OPENS', '1')
    module.enable_open_diagnostics()
    hooks[0]('open', ('/tmp/example.pyc', 'r', 0))
    hooks[0]('import', ('unused',))
    output = capsys.readouterr()
    assert 'PYTHON_OPEN' in output.err and '/tmp/example.pyc' in output.err
    assert 'unused' not in output.err
