"""Paste into the existing Colab session after approved profile and budget review."""
import json
import os
import shutil
from cloud_workflow import prepare_plan, seal_profile, run_arm

os.chdir(project)
study_name = "cuda_jpeg_5000_v1"
plan_dir = project / "reports" / study_name
plan_file = prepare_plan(project, plan_dir, max_seconds=9000)
reviewed_plan = seal_profile(project, plan_file, profile_file)

plan = json.loads(reviewed_plan.read_text())
plan["cloud_policy"]["long_run_approved"] = True
reviewed_plan.write_text(json.dumps(plan, indent=2) + "\n")

backup_root = drive_root / "studies"
study_backup = backup_root / study_name
study_backup.mkdir(parents=True, exist_ok=False)
shutil.copy2(reviewed_plan, study_backup / "reviewed_plan.json")
shutil.copy2(profile_file, study_backup / "cuda_profile.json")
approval = {
    "date_asia_kolkata": "2026-10-22",
    "user_message": "yeah approve cloud study",
    "arms": ["A_seed42", "B_seed42", "C_seed42"],
    "updates_per_arm": 5000,
    "seconds_per_arm_invocation": 9000,
    "total_training_seconds": 27000,
    "spending_usd": 0,
    "downloads": False,
    "protected_test_scoring": False,
    "automatic_restarts": False,
}
(study_backup / "approval.json").write_text(json.dumps(approval, indent=2) + "\n")
APPROVED_LONG_RUN = True
print("PLAN READY — saved to", study_backup)
