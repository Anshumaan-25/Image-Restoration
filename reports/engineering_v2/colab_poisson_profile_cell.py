from google.colab import drive, files
from pathlib import Path
import hashlib, subprocess, sys, shutil

drive.mount('/content/drive')
print('Select BOTH colab_extension_v2.zip and colab_poisson_launcher.py from Downloads.')
uploaded = files.upload()
expected = {'colab_extension_v2.zip': 'f3665a079fb7c028244a8df6cbf4b6f9bb1808834bb1ad6e15fd9e944550024b', 'colab_poisson_launcher.py': 'bbbe6ffdf8490337cb1682facec1296c12857b288f36f3d00f82fca9329424be'}
for name, digest in expected.items():
    assert name in uploaded, f'Missing {name}'
    assert hashlib.sha256(uploaded[name]).hexdigest() == digest, f'Hash mismatch: {name}'
    Path('/content', name).write_bytes(uploaded[name])
backup = Path('/content/drive/MyDrive/RestorationProject/colab_extension_v2.zip')
if backup.exists():
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == expected['colab_extension_v2.zip']
else:
    shutil.copyfile('/content/colab_extension_v2.zip', backup)
with subprocess.Popen([sys.executable, '/content/colab_poisson_launcher.py', '--mode', 'profile'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end='', flush=True)
    code = process.wait()
if code:
    raise RuntimeError('Profile stopped. Share the output; no training was started.')
