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
