# Free Colab continuation

The user requested that remaining long training move off the Mac on 2026-10-21.
The approved v1.0 architecture is unchanged. Use **Colab Free**, one allocated CUDA
GPU, $0. Existing MPS results remain valid local evidence. The new matched CUDA
study must train **A, B and C from scratch in the same environment**; combining
new CUDA/FP16 A/C with old MPS/FP32 B would confound the method comparison.
No additional long run is authorized merely by preparing this workflow. For the current session, the user separately approved the [measured three-arm budget](../reports/proposals/colab_matched5000_v1/approval.json) on 2026-10-22: 5,000 updates and 9,000 seconds per arm, 27,000 seconds combined, $0. [Preparation cell](../reports/proposals/colab_matched5000_v1/prepare_cell.py) and [A launch cell](../reports/proposals/colab_matched5000_v1/launch_A_cell.py) replace the original notebook's old 3,600-second proposal. All runtime profile/source/environment checks still apply.

## Start here

1. Open [Google Colab](https://colab.research.google.com/), choose Upload notebook,
   and upload [free_gpu.ipynb](../notebooks/free_gpu.ipynb).
2. In your private Google Drive, create `RestorationProject` and upload
   `dist/colab_restoration_v1.zip`. The local `.sha256` file identifies the exact
   transfer. The ZIP is approximately 176 MiB: code plus **32 training / 8
   selection-development PNGs**, not all 900 images. No weights or protected-test
   images are included. Keep the transfer private; this is not a public dataset release.
3. Select a free GPU runtime and connect. The notebook shows the actual GPU;
   a T4 is a target, not an allocation guarantee. Authorize the explicit Drive
   mount in the notebook. No cloud-account access is performed from the Mac.
4. Run setup, integrity checks, offline tests and the two-update CUDA smoke test.
   The notebook retains preinstalled compatible PyTorch; it does not apply the
   Mac environment lock or automatically install/reinstall packages.
5. Generate the provisional matched plan and run the bounded actual-workload
   profile: 10 warmups + 50 measured updates, at most 300 seconds plus setup/save.
   CUDA FP16, 128x128, **clean microbatch 2 / accumulation 4**, two views. The BN
   forward batch contains four views; accumulation does not change that BN batch.
6. Inspect completed profile, numerical failures/skips, peak allocated/reserved
   memory, actual environment and reviewed plan. Target allocated <=12 GiB and
   reserved <=95% of the allocated GPU's total memory. No CUDA timing or fit is
   claimed before this succeeds. Profile timing excludes validation/backup;
   its 15,000-update projection is only a lower-scope cost estimate. A projection
   that already exhausts the cap fails; retain it and prepare a new common plan
   with a feasible budget before approving any arm. Passing this check does not
   prove that validation/backup overhead fits.
7. Review the concrete long-run budget before explicitly setting the notebook's
   `APPROVED_LONG_RUN` flag. Default is false, so Run all stops before research.
   Provisional budget: **3 x 5,000 successful updates, seed 42, 3,600 seconds per
   invocation / 10,800 seconds combined**, plus initial/final I/O. These are soft
   optimizer/validation-boundary caps, not promises of completion. If infeasible,
   revise one common schedule before training any arm; do not shorten only C.
8. Run A, B and C cells in order. An incomplete arm raises an error; stop there.
   No protected scoring, pretrained downloads, paid upgrade or automatic seed
   expansion occurs. Review development effects and prior art before later tests.

## Checkpoints, interruptions and recovery

Training data and active runs live on `/content` local storage, avoiding small
Drive reads/writes each update. A thin helper calls the existing trainer and
saves versioned, integrity-checked private Drive ZIP snapshots after the first
100-update checkpoint, then roughly every 200 successful updates, plus normal
completion/error exit. Both latest and selected best state are preserved; logs
are trimmed to the saved checkpoint's microstep. The helper is separately hashed
in the plan. Training/core/config source files remain unchanged.

The callback pauses training while copying a snapshot; its synchronization and
transfer overhead apply equally to all arms and consume runtime. A consistency check refuses
mixed best/latest states. Snapshots are built locally, copied as `.partial`, then
renamed and validated; restore verifies CRC/hashes before use. Retain older snapshots
and ignore incomplete `.partial` files. Drive failures raise an error; there is no
silent continuation without preservation. Existing snapshots are never deleted.
Allow roughly **2–3 GB free Drive headroom** for the input bundle and retained
snapshots of this small three-arm study; actual use depends on interruptions.

An abrupt free-runtime termination can lose updates since the last completed
snapshot. This is interruption tolerance, not guaranteed persistence or exact
CUDA replay. For recovery, unpack the original bundle on a new free VM, retain
the original reviewed plan/profile, and restore a verified ZIP using the recovery
cell. The environment/codec/GPU identity and original schedule must match; a
changed allocation fails for review. Only resume the interrupted arm, then its
remaining controls. **A resumed invocation requires reviewing remaining wall-time
and approval again**; the cap is per invocation, not an unlimited retry allowance.
Do not restart a completed arm or reuse the old Mac B as a CUDA control.

Run summaries, checkpoint source/data identities and scalar logs remain the
authoritative results. Before final reporting, independently replay selected dev
checkpoints and matched comparisons using the existing evaluation scripts. This
notebook does not establish candidate benefit, convergence or publishable novelty.

## Queue approved B/C while A is running

The operator explicitly requested automatic next-arm execution while AFK. Add the
[continuation cell](../reports/proposals/colab_matched5000_v1/queue_BC_cell.py)
immediately below the running A cell and submit only that new cell for execution.
Confirm the UI shows it waiting/queued; cancel any prompt offering to interrupt A.
Keep the browser/runtime connected. Do not use Run all, rerun A/preparation, or
edit the trainer/helper during training. A successful return assigns results[A]
before the queued cell checks it; failure/incomplete A blocks all continuation.
The cell runs B then C under the existing 9,000-second caps, stops on any error
or incomplete outcome, and rejects duplicate execution once B/C results exist.
No auto-retry, CPU fallback, new seed, extra run or quota bypass is added.

All three arms already use verified, versioned Drive checkpoints about every 200
updates and on normal/handled exits. A quota/VM kill may give no warning and may
not execute finally; recovery is from the last complete ZIP, with roughly the
intervening 200 updates potentially lost. It cannot guarantee a final save, another
GPU allocation or uninterrupted AFK execution. Existing source/environment/config
checks still apply to later manual recovery; no schedule reset is authorized.

The user's Mac can remain plugged in with its lid open and the Colab tab open.
Running `caffeinate -i -t 28800` in the Mac's Terminal prevents local idle sleep
for eight hours (Ctrl+C ends it sooner); it does not alter Colab limits. This is
an optional operator action, not a locally launched training process. Do not use
automated clicks or reconnection loops to try to bypass free-tier limits.

[Offline dispatch verification](../reports/proposals/colab_matched5000_v1/continuation_validation.json)
passed four coordinator cases and the required 47-test core acceptance/smoke/
syntax/two-update CPU dry-run/dependency checks; all frozen training-source hashes
match. Actual Colab frontend queue behavior requires operator confirmation.
The last user-reported A backup is update 2,700, microstep 10,800 (54%); A completion
and actual B/C starts are not inferred from preparing this cell.

## Resource and environment limits

Colab Free GPU types, quotas and session duration vary; free sessions can end
early. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).
There is no assumption of unlimited free compute or an automatic paid fallback.
Kaggle Notebooks is another free option if an accelerator is offered to the
account; its setup/storage UI differs. This prepared notebook specifically uses
Colab Drive mounting. Two Kaggle GPUs have separate memories, not pooled VRAM;
this project uses one GPU and needs no distributed rewrite.

The Mac performs only packaging and short CPU QA for this change. No further
local research training is started. Actual CUDA/FP16 behavior and Google Drive
mount/storage operations must still be verified on the allocated cloud VM.

## Existing v1 upload: setup correction (2026-10-22)

A clean v1 archive omits the empty `reports/` directory. Two cloud tests used it
as a temporary-directory parent and failed before planning. This was reproduced
from the original ZIP; it is not evidence of a GPU or package incompatibility.
The existing upload needs only `(project / 'reports').mkdir(parents=True,
exist_ok=True)` before rerunning the cloud tests. Keep the original images/ZIP and
completed core checks. No model/source/data edits, reinstall or runtime reset are
needed for this missing-folder issue.

The current local notebook initializes the directory during setup and QA, and
streams child stdout/stderr so failures include their original cause. Updated
local fixture tests no longer assume `reports/` exists. Three notebook regression
checks cover directory initialization, visible successful output and fatal errors
with retained diagnostics. [Reproduction and repair evidence](../reports/colab_setup_fix_v1.json)
and [fresh core acceptance](../reports/colab_setup_fix_core_acceptance_v1.json) retain
the original failure and successful local verification. The v1 archive/old notebook
hash records are historical; the corrected external notebook has a new hash while
the input ZIP and all frozen training-source hashes stay unchanged. Actual Colab
QA/FP16 profiling remains pending the user's rerun.

## Developer verification

```bash
python notebooks/test_cloud_workflow.py
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_test.py
python -m py_compile train.py eval.py inference.py models/*.py losses/*.py utils/*.py notebooks/*.py
python train.py --dry-run --epochs 1 --batch-size 2 --device cpu
python notebooks/prepare_colab.py --output dist/NEW_colab_bundle.zip
```

Packaging rejects image hash/identity/split failures and existing output ZIPs.
Tests cover traversal/symlinks/duplicate ZIP names, transfer corruption, the
protected-data guard, matched plan and approval/device failures, best/latest
consistency, trimmed logs, corruption before extraction and exact CPU resume
through a real snapshot. Cloud helpers/tests live under notebooks so existing
research checkpoint source identities remain unchanged.
