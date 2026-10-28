from google.colab import drive
from pathlib import Path
import hashlib, subprocess, sys, shutil

drive.mount('/content/drive')
assert Path('/content/drive').is_mount(), 'Drive must be durably mounted'
source = Path('/content/drive/MyDrive/RestorationProject')
expected = {'colab_extension_v2.zip': 'f3665a079fb7c028244a8df6cbf4b6f9bb1808834bb1ad6e15fd9e944550024b', 'colab_poisson_launcher.py': 'bbbe6ffdf8490337cb1682facec1296c12857b288f36f3d00f82fca9329424be'}
for name, digest in expected.items():
    path = source / name
    assert path.is_file(), f'Wait for Drive upload completion: {name}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Hash mismatch: {name}'
    shutil.copyfile(path, Path('/content') / name)
print('Both Drive artifacts verified. Profile first; approved training only if all gates pass.', flush=True)
for mode, extra in [('profile', []), ('train', ['--approved-training'])]:
    print('Starting approved stage:', mode, flush=True)
    with subprocess.Popen([sys.executable, '/content/colab_poisson_launcher.py', '--mode', mode, *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'{mode} stopped. Retain outputs; no automatic retry or fallback.')
