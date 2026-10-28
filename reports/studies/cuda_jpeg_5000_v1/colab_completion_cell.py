from google.colab import drive, files
from pathlib import Path
import hashlib, subprocess, sys

drive.mount('/content/drive')
print('Select colab_completion_checks.py from your Mac Downloads folder.')
uploaded = files.upload()
name = 'colab_completion_checks.py'
assert name in uploaded, 'Please select only colab_completion_checks.py.'
assert hashlib.sha256(uploaded[name]).hexdigest() == 'abcefa1b0e9cdd8bf86c0fbcac3929d5a3e8d0f62e89833fc2de3f5faa23788a', 'Launcher hash mismatch; do not run.'
launcher = Path('/content/colab_completion_checks.py')
launcher.write_bytes(uploaded[name])
with subprocess.Popen([sys.executable, str(launcher), '--approved'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end='', flush=True)
    code = process.wait()
if code:
    raise RuntimeError('Checks stopped. Share the error; do not rerun this cell blindly.')
