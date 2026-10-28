# APPROVED OPTIMIZED SIDD VALIDATION ONLY — no training
from pathlib import Path
import hashlib,shutil,subprocess,sys
assert Path('/content/drive').is_mount(), 'Drive must remain mounted'
p = Path('/content/drive/MyDrive/RestorationProject/colab_sidd_full_validation_optimized_v3.py')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '6960ca00d159b57fe2386a3d998aaad96c71ac1d440883f2beaae91e135d3c58'
local=Path('/content/colab_sidd_full_validation_optimized_v3.py')
shutil.copy2(p,local)
with subprocess.Popen([sys.executable,str(local),'--approved-full-validation'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
    for line in process.stdout:
        print(line,end='',flush=True)
    code=process.wait()
if code:
    raise RuntimeError('Optimized SIDD stopped; preserve all evidence; no automatic retry.')
