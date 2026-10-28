from google.colab import drive
from pathlib import Path
import hashlib, subprocess, sys, shutil

drive.mount('/content/drive')
assert Path('/content/drive').is_mount(), 'Drive must be durably mounted'
source = Path('/content/drive/MyDrive/RestorationProject')
expected = {'colab_extension_v3.zip': 'db81ac9de44a48699de33c00d5dba16aa476f11f376f797a8f0bee38ee8c2240', 'colab_poisson_launcher_v3.py': '71790fa7380d06aa2c826b2be901a7c20b4c14b4516733c2432e33325612a9a4'}
for name, digest in expected.items():
    path = source / name
    assert path.is_file(), f'Wait for Drive upload completion: {name}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Hash mismatch: {name}'
    shutil.copyfile(path, Path('/content') / name)
print('Both Drive artifacts verified. Profile first; approved training only if all gates pass.', flush=True)
for mode, extra in [('profile', []), ('train', ['--approved-training'])]:
    print('Starting approved stage:', mode, flush=True)
    with subprocess.Popen([sys.executable, '/content/colab_poisson_launcher_v3.py', '--mode', mode, *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'{mode} stopped. Retain outputs; no automatic retry or fallback.')
