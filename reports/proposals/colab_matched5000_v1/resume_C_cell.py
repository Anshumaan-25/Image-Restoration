"""Colab C-only recovery after explicit approval of this resumed invocation."""
from pathlib import Path
import json
import os
from cloud_workflow import run_arm, sha256

project = Path('/content/Image_Degradatation_Restoration')
os.chdir(project)
drive_root = Path('/content/drive/MyDrive/RestorationProject')
backup_root = drive_root / 'studies'
study = backup_root / 'cuda_jpeg_5000_v1'
reviewed_plan = study / 'reviewed_plan.json'
resume_archive = study / 'C_seed42' / 'step_003700_micro_00014800.zip'
plan = json.loads(reviewed_plan.read_text())
profile_source = study / 'cuda_profile.json'
assert sha256(profile_source) == plan['cuda_profile']['sha256']
profile_target = project / plan['cuda_profile']['path']
assert profile_target.resolve().is_relative_to(project.resolve())
profile_target.parent.mkdir(parents=True, exist_ok=True)
if profile_target.exists():
    assert sha256(profile_target) == plan['cuda_profile']['sha256']
else:
    profile_target.write_bytes(profile_source.read_bytes())
assert sha256(profile_target) == plan['cuda_profile']['sha256']

# Enable only after explicit approval of the C-only recovery invocation.
APPROVED_C_RESUME = True
if APPROVED_C_RESUME is not True:
    raise PermissionError('C-only resume invocation requires approval')
result_C = run_arm(
    project, reviewed_plan, 'C_seed42', backup_root,
    approved=APPROVED_C_RESUME, resume_archive=resume_archive,
)
print(json.dumps(result_C, indent=2))
