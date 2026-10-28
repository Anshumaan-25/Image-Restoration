"""Paste below the successful preparation cell; run exactly once for arm A."""
assert APPROVED_LONG_RUN
results = {}
results["A"] = run_arm(project, reviewed_plan, "A_seed42", backup_root,
                       approved=APPROVED_LONG_RUN)
print(json.dumps(results["A"], indent=2))
