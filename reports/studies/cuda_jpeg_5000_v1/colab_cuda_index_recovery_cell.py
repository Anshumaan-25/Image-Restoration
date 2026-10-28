from pathlib import Path
import hashlib, subprocess, sys

source = Path('/content/colab_completion_checks.py')
assert source.is_file(), 'Original launcher missing; share this message.'
original = source.read_bytes()
assert hashlib.sha256(original).hexdigest() == '845643531cdf037c4b55b7091972d80b422347177efd943c9f72666341a7ce9e', 'Unexpected launcher; stop.'
fixed = original.decode()
for before, after in [('torch.cuda.set_per_process_memory_fraction(min(1.0,12*1024**3/total),device)', 'torch.cuda.set_per_process_memory_fraction(min(1.0,12*1024**3/total),torch.cuda.current_device() if device.index is None else device.index)'), ("destination=study/'completion_checks_v1'", "destination=study/'completion_checks_v1_cuda_index_fix'"), ("export=study/'restoration_completion_checks_v1.zip'", "export=study/'restoration_completion_checks_v1_cuda_index_fix.zip'")]:
    assert fixed.count(before) == 1
    fixed = fixed.replace(before, after)
assert hashlib.sha256(fixed.encode()).hexdigest() == 'abcefa1b0e9cdd8bf86c0fbcac3929d5a3e8d0f62e89833fc2de3f5faa23788a'
launcher = Path('/content/colab_completion_checks_cuda_index_fix.py')
if launcher.exists():
    assert launcher.read_text() == fixed, 'Recovery file differs; stop.'
else:
    launcher.write_text(fixed)
print('CUDA device-index fix verified. Original logs will be retained.')
with subprocess.Popen([sys.executable, str(launcher), '--approved'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
    for line in process.stdout:
        print(line, end='', flush=True)
    code = process.wait()
if code:
    raise RuntimeError('Checks stopped; share the output before rerunning.')
