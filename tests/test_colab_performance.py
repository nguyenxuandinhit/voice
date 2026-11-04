"""Exercise the notebook patch/preset cells without a Colab account or models."""
import ast
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/OmniVoice_Studio_Colab.ipynb'


def source(marker):
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    return next(''.join(c['source']) for c in cells if marker in ''.join(c['source']))


@pytest.mark.parametrize('state', ['applied', 'new', 'conflict', 'absent'])
def test_local_patch_is_idempotent_and_never_resets_checkout(state, monkeypatch):
    code = source('# 2b. System packages')
    block = code[code.index('# Optional local performance patch.'):code.index('# 2b. System packages')]
    commands = []
    monkeypatch.setattr(Path, 'is_file', lambda self: state != 'absent')
    def check(argv, **kwargs):
        commands.append(argv)
        ok = (state == 'applied' and '--reverse' in argv) or (state == 'new' and '--reverse' not in argv)
        return types.SimpleNamespace(returncode=0 if ok else 1)
    def run(argv, **kwargs):
        commands.append(argv)
    ns = {'subprocess': types.SimpleNamespace(run=check), 'REPO_DIR': '/repo', 'run': run}
    if state == 'conflict':
        with pytest.raises(SystemExit, match='does not match'):
            exec(block, ns)
    else:
        exec(block, ns)
    writes = [cmd for cmd in commands if '--check' not in cmd]
    assert len(writes) == (1 if state == 'new' else 0)
    assert all('reset' not in cmd and 'stash' not in cmd for cmd in commands)


def test_notebook_selects_profile_before_launch_and_clears_stale_beam(monkeypatch):
    code = source('# Explicit performance preset')
    start = code.index('# Explicit performance preset')
    block = code[start:code.index('info = health()', start)]
    import os
    monkeypatch.setattr(os, 'environ', {'OMNIVOICE_GPU_WORKERS': '4', 'OMNIVOICE_API_KEY': 'secret'})
    exec(block, {'REPO_DIR': str(ROOT), 'os': os})
    assert os.environ['OMNIVOICE_GPU_WORKERS'] == '1'
    assert os.environ['OMNIVOICE_API_KEY'] == 'secret'
    os.environ['OMNIVOICE_INDEXTTS_NUM_BEAMS'] = '1'
    exec(block, {'REPO_DIR': str(ROOT), 'os': os})
    assert 'OMNIVOICE_INDEXTTS_NUM_BEAMS' not in os.environ
    assert code.index('# Explicit performance preset') < code.index('subprocess.Popen(')


def test_missing_patch_has_an_actionable_message():
    import os
    code = source('# Explicit performance preset')
    block = code[code.index('# Explicit performance preset'):code.index('info = health()')]
    with pytest.raises(SystemExit, match='Upload voicestudio-performance.patch'):
        exec(block, {'REPO_DIR': str(ROOT / '.tmp/nonexistent-repo'), 'os': os})


def test_all_notebook_python_cells_parse():
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    for cell in cells:
        if cell['cell_type'] == 'code':
            # IPython shell escapes are valid notebook syntax, not Python AST.
            code = ''.join(line for line in cell['source'] if not line.lstrip().startswith(('!', '%')))
            ast.parse(code)


@pytest.mark.parametrize('existing,matching', [(False, True), (True, True), (True, False)])
def test_pinned_setup_reuses_modified_checkout_without_pulling_or_resetting(existing, matching):
    code = source('# 2b. System packages')
    block = code[code.index('# 2a. Clone the repo'):code.index('# Optional local performance patch.')]
    revision = 'a' * 40
    calls = []
    fake_subprocess = types.SimpleNamespace(run=lambda *a, **kw: types.SimpleNamespace(stdout=(revision if matching else 'b' * 40) + '\n'))
    ns = {
        'os': types.SimpleNamespace(path=types.SimpleNamespace(isdir=lambda path: existing, join=lambda *p: '/'.join(p))),
        'subprocess': fake_subprocess,
        'shutil': types.SimpleNamespace(rmtree=lambda *a, **kw: pytest.fail('must not delete files')),
        'REPO_DIR': '/repo', 'REPO_URL': 'https://github.com/debpalash/VoiceStudio',
        'REPO_REVISION': revision, 'run': lambda argv, **kw: calls.append(argv),
    }
    if existing and not matching:
        with pytest.raises(SystemExit, match='no files were reset'):
            exec(block, ns)
        assert not calls
    else:
        exec(block, ns)
        assert len(calls) == (0 if existing else 3)
        assert not any('pull' in argv or 'reset' in argv for argv in calls)
