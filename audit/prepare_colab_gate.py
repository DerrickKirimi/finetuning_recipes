"""Package exact local environment inputs and generate a private Colab execution notebook."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
import zipfile

from assets import output_directory


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {},
              'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = output_directory(args.output) / stamp
    out.mkdir()
    names = ['pyproject.toml', 'uv.lock', 'README.md', 'check_env.sh',
             'audit/hardware.py', 'audit/assets.py', 'audit/colab_gate.py']
    payload = {name: (root / name).read_bytes() for name in names}
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root, text=True).strip()
    manifest = {'created_utc': stamp, 'git_head': git('rev-parse', 'HEAD'),
                'git_branch': git('branch', '--show-current'),
                'scope': 'Environment-only subset of local working tree; not a complete training checkout',
                'git_status': git('status', '--short'),
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
                'uv_version': '0.8.13', 'python_version': '3.12.10'}
    archive = out / f'colab-env-inputs-{stamp}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in payload.items():
            z.writestr(name, data)
        z.writestr('bundle-manifest.json', json.dumps(manifest, indent=2) + '\n')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / 'bundle-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'accelerator': 'GPU', 'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'},
        'language_info': {'name': 'python'}}, 'cells': []}
    notebook['cells'].append(cell('markdown', f'''
        # Verify the locked Colab environment

        Open this local notebook in VS Code. Select the existing Colab T4 server and
        its Python 3 kernel. Upload `{archive.name}` to `/content` with Upload to Colab.
        Run the next two code cells in order; save with Ctrl+S after each.
        The second cell downloads and installs the project dependencies into a fresh
        Python 3.12.10 virtual environment, then verifies GPU execution and imports.
        The notebook kernel remains Colab's preinstalled Python and invokes the new
        environment through subprocesses. No model weights, training or teacher jobs run.
        Several GB of dependency downloads are expected. Stop at a failure and save it.
    '''))
    setup = '''
        import hashlib, json, os, shutil, subprocess, sys, zipfile
        from datetime import datetime, timezone
        from pathlib import Path
        import torch

        assert torch.cuda.is_available(), 'Select a Colab GPU server first.'
        archive = Path('/content') / ARCHIVE_NAME
        assert archive.is_file(), f'Upload {archive.name} to /content first.'
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_HASH, 'Archive hash mismatch.'
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        work = Path('/content/posttraining-env-runs') / run_id
        project, results = work / 'code', work / 'results'
        project.mkdir(parents=True)
        results.mkdir()
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('bundle-manifest.json'))
            assert set(z.namelist()) == set(manifest['files']) | {'bundle-manifest.json'}
            for name, expected in manifest['files'].items():
                dest = (project / name).resolve()
                assert dest.is_relative_to(project.resolve()), name
                data = z.read(name)
                assert hashlib.sha256(data).hexdigest() == expected, name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            (project / 'bundle-manifest.json').write_text(json.dumps(manifest, indent=2))
        print('GPU:', torch.cuda.get_device_name(0))
        print('Native BF16:', torch.cuda.is_bf16_supported(including_emulation=False))
        print('Fresh run directory:', work)
        print('Bundle verified:', ARCHIVE_HASH)
    '''
    setup = setup.replace('ARCHIVE_NAME', repr(archive.name)).replace('ARCHIVE_HASH', repr(digest))
    notebook['cells'].append(cell('code', setup))
    notebook['cells'].append(cell('code', '''
        command = [sys.executable, '-u', str(project / 'audit/colab_gate.py'),
                   '--project', str(project), '--output', str(results)]
        print('Starting locked install and checks. Save this notebook when finished.', flush=True)
        with (results / 'driver.log').open('w') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                exit_code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        for name in ['gate.json', 'environment.json']:
            if (results / name).exists():
                print(f'\\n{name}\\n' + (results / name).read_text())
        assert exit_code == 0, 'Environment gate failed. Save all outputs and return for review.'
        print('ENVIRONMENT GATE PASSED. CPT has not run. Save with Ctrl+S.')
    '''))
    notebook['cells'].append(cell('markdown', '''
        ## Optional: persist logs to Drive, including after a failed check

        Save the notebook first. The runner already copies reports when Drive is mounted.
        This cell mounts Drive if necessary and copies only this run's results, not its
        virtual environment or dependencies. Run it separately, even after a check error.
    '''))
    notebook['cells'].append(cell('code', '''
        from google.colab import drive
        if not Path('/content/drive/MyDrive').is_dir():
            drive.mount('/content/drive')
        destination = Path('/content/drive/MyDrive/posttraining-results/colab-locked-env') / run_id
        shutil.copytree(results, destination, dirs_exist_ok=True)
        print('Persistent results:', destination)
    '''))
    for i, c in enumerate(notebook['cells']):
        c['id'] = f'gate-{i}'
    (out / 'locked-environment.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    print(json.dumps({'directory': str(out), 'archive': str(archive), 'sha256': digest,
                      'notebook': str(out / 'locked-environment.ipynb')}, indent=2))


if __name__ == '__main__':
    main()
