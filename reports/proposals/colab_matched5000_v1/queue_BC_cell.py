"""Queue this cell below running A; continue approved B/C only after A succeeds."""
if APPROVED_LONG_RUN is not True:
    raise RuntimeError("The bounded study must be approved before continuing.")
if results.get("A", {}).get("completed") is not True:
    raise RuntimeError("A did not complete; B and C will not start.")
if any(arm in results for arm in ("B", "C")):
    raise RuntimeError("B/C already have results; do not rerun this continuation cell.")

for arm in ("B", "C"):
    print(f"Starting arm {arm}", flush=True)
    outcome = run_arm(project, reviewed_plan, f"{arm}_seed42", backup_root,
                      approved=APPROVED_LONG_RUN)
    if outcome.get("completed") is not True:
        raise RuntimeError(f"Arm {arm} is incomplete; stopping the study.")
    results[arm] = outcome
    print(json.dumps(outcome, indent=2), flush=True)

print("ALL THREE ARMS COMPLETED", flush=True)
