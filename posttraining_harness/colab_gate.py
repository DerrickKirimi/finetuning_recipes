"""Verify a fresh locked CUDA environment; run only on a Colab GPU server."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if not Path('/content').is_dir() or not shutil.which('nvidia-smi'):
        raise SystemExit('Run this check on the Colab GPU server, not locally.')
    if output.is_relative_to(project):
        raise SystemExit('Outputs must be outside the code directory.')
    output.mkdir(parents=True, exist_ok=True)
    status = {'pass': False, 'scope': 'Locked environment and imports only; CPT NOT RUN',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'commands': []}
    env = os.environ.copy()
    env.pop('VIRTUAL_ENV', None)
    env.pop('PYTHONPATH', None)
    env.update(UV_PROJECT_ENVIRONMENT=str(project / '.venv'),
               UV_CACHE_DIR=str(project.parent / 'uv-cache'),
               UV_PYTHON_INSTALL_DIR=str(project.parent / 'python'),
               PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               HF_HOME=str(project.parent / 'hf-cache'), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled')

    def save():
        (output / 'gate.json').write_text(json.dumps(status, indent=2) + '\n')
        # Copy only reports/logs when Drive is already mounted; never copy environments.
        drive = Path('/content/drive/MyDrive')
        if drive.is_dir():
            try:
                dest = drive / 'posttraining-results' / 'colab-locked-env' / output.parent.name
                shutil.copytree(output, dest, dirs_exist_ok=True)
            except OSError as exc:
                print('Drive backup failed; local notebook output remains available:', exc, flush=True)

    def run(name, command):
        print(f'\n>>> {name}: {command}', flush=True)
        entry = {'name': name, 'argv': list(map(str, command)), 'cwd': str(project),
                 'exit_code': None, 'log': name + '.log'}
        status['commands'].append(entry)
        save()
        with (output / entry['log']).open('w') as log:
            proc = subprocess.Popen(entry['argv'], cwd=project, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                entry['exit_code'] = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                save()
        if entry['exit_code']:
            raise RuntimeError(f'{name} failed with exit code {entry["exit_code"]}; inspect its log.')

    try:
        assert not (project / '.venv').exists(), 'Fresh environment required; rerun notebook preparation cell.'
        status['disk_free_gib_before'] = round(shutil.disk_usage(project).free / 2**30, 2)
        assert status['disk_free_gib_before'] >= 30, 'Need at least 30 GiB free for this install attempt.'
        manifest = json.loads((project / 'bundle-manifest.json').read_text())
        status['bundle_manifest'] = manifest
        for name, digest in manifest['files'].items():
            assert hashlib.sha256((project / name).read_bytes()).hexdigest() == digest, name
        tools = project.parent / 'bootstrap'
        run('bootstrap-uv', [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                            '--no-cache-dir', '--target', tools, 'uv==0.8.13'])
        uv = tools / 'bin' / 'uv'
        run('uv-version', [uv, '--version'])
        run('sync', [uv, 'sync', '--locked', '--python', '3.12.10'])
        assert hashlib.sha256((project / 'uv.lock').read_bytes()).hexdigest() == manifest['files']['uv.lock']
        python = project / '.venv/bin/python'
        env['PY'] = str(python)
        run('hardware', ['bash', 'check_env.sh', '--verify', '--require-cuda', '--output', output])
        run('imports', [python, '-c', '''
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
import torch, transformers, trl, datasets, peft, xformers, bitsandbytes
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
import json, sys, importlib.metadata as md
assert sys.version_info[:3] == (3, 12, 10), sys.version
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert transformers.__version__ == '4.57.6', transformers.__version__
assert trl.__version__ == '0.24.0', trl.__version__
assert torch.cuda.is_available()
print(json.dumps({name: md.version(name) for name in
    ['torch', 'transformers', 'trl', 'datasets', 'peft', 'unsloth', 'unsloth-zoo',
     'xformers', 'bitsandbytes', 'neural-txt', 'text-albumentations']}, indent=2))
'''])
        hardware = json.loads((output / 'environment.json').read_text())
        assert hardware['pass']
        if any(d['capability'][0] < 8 for d in hardware['devices']):
            assert hardware['profile']['dtype'] == 'float16', hardware['profile']
        status['pass'] = True
    except Exception as exc:
        status['error'] = str(exc)
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print('\nGATE RESULT\n' + json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
