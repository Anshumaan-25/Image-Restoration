# Run this ONE NEW cell in the browser session where Google Drive mounting works.
# Approvals already recorded: 900 Poisson updates/15min, then SIDD1280/20min.
from google.colab import drive
from pathlib import Path
import hashlib, shutil, subprocess, sys

drive.mount('/content/drive')
assert Path('/content/drive').is_mount(), 'Google Drive must be mounted'
DRIVE_PROJECT = Path('/content/drive/MyDrive/RestorationProject')
FILES = {
    'colab_poisson_recovery_preflight_cell.py': 'b9ceb05713861c82fbfec732e1226622a0a96d771b2984b8e10346daa6f05426',
    'colab_finish_approved_stages.py': '4c31b8e243e6148ad2a1c2eede65bd12df8c1437f31fb8d97b9ac80bd90d2bf8',
}
for name, digest in FILES.items():
    file = DRIVE_PROJECT / name
    assert file.is_file(), f'Missing uploaded helper: {name}'
    assert hashlib.sha256(file.read_bytes()).hexdigest() == digest, f'Helper hash mismatch: {name}'

# Inspection only: restores the frozen source, checks the snapshot and environment.
file = DRIVE_PROJECT / 'colab_poisson_recovery_preflight_cell.py'
exec(compile(file.read_bytes(), str(file), 'exec'), globals())

# Each approved stage checks its own state, resource limits and prerequisites.
file = DRIVE_PROJECT / 'colab_finish_approved_stages.py'
local = Path('/content/colab_finish_approved_stages.py')
shutil.copyfile(file, local)
with subprocess.Popen([sys.executable, str(local), '--approved-stages'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end='', flush=True)
    code = process.wait()
if code:
    raise RuntimeError('A stage stopped. Keep all outputs and share the error; no automatic retry.')
