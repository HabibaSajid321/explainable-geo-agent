"""
Human Review Agent
-------------------
The trust layer GeoEvolve doesn't have: a gate where a human expert sees
the selected model, its rationale, and its calibration numbers, then
approves, rejects, or requests re-selection with a note -- logged to an
append-only audit trail. Same pattern as the approval workflow in the SOC
assistant / bid-automation platform, applied to model selection instead
of remediation actions.

`auto_approve_threshold`: if empirical coverage is at or above this and
within `coverage_tolerance` of target, auto-approve (useful for batch /
CI runs). Otherwise this pauses for interactive input.
"""
import json
import time
from pathlib import Path

AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "audit_log.jsonl"


def _log(entry: dict):
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def review(discovery_out, eval_out, interactive=False, coverage_tolerance=0.05):
    target = eval_out["target_coverage"]
    actual = eval_out["empirical_coverage"]
    current_scaling = eval_out.get("scaling_factor", 1.0)
    within_tolerance = abs(actual - target) <= coverage_tolerance

    recalibrate_needed = not within_tolerance
    # Calculate suggested conformal scaling adjustment
    if actual > 0:
        suggested_scaling = round(current_scaling * (target / actual), 3)
    else:
        suggested_scaling = round(current_scaling * 1.2, 3)

    decision, note = None, ""
    if interactive:
        print("\n--- Human Review ---")
        print(f"Selected model : {discovery_out['selected_model']}")
        print(f"Rationale      : {discovery_out['rationale']}")
        print(f"Target coverage: {target*100:.0f}%  |  Empirical: {actual*100:.1f}%")
        resp = input("Approve this model? [y/n/recalibrate/comment]: ").strip()
        if resp.lower() in ("y", "yes"):
            decision = "approved"
            recalibrate_needed = False
        elif resp.lower() in ("n", "no"):
            decision = "rejected"
            recalibrate_needed = False
        elif resp.lower() in ("r", "recalibrate"):
            decision = "recalibrate_requested"
            recalibrate_needed = True
        else:
            decision, note = "flagged", resp
    else:
        decision = "approved" if within_tolerance else "flagged_for_recalibration"
        note = "" if within_tolerance else (
            f"Empirical coverage {actual*100:.1f}% deviates from target "
            f"{target*100:.0f}% by more than {coverage_tolerance*100:.0f}pp."
        )

    entry = {
        "timestamp": time.time(),
        "model": discovery_out["selected_model"],
        "cv_scores": discovery_out["scores"],
        "rmse": eval_out["rmse"],
        "target_coverage": target,
        "empirical_coverage": actual,
        "decision": decision,
        "note": note,
        "recalibrate_needed": recalibrate_needed,
        "suggested_scaling_factor": suggested_scaling,
    }
    _log(entry)
    return entry

