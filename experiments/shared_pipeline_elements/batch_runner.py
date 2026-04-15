"""
batch_runner.py
───────────────
Shared batch execution, result persistence, predictions export,
summary stats, and CLI entry point.

All three technique scripts use this identically — they just pass in
their pipeline instance:

    # At the bottom of generate_validate_retry.py:
    if __name__ == "__main__":
        from batch_runner import main
        main(GVRPipeline(), default_results_path="results/gvr_results.json")

    # At the bottom of prompt_engineering.py:
    if __name__ == "__main__":
        from batch_runner import main
        main(PromptEngineeringPipeline(), default_results_path="results/pe_results.json")

    # At the bottom of constrained_decoding.py:
    if __name__ == "__main__":
        from batch_runner import main
        main(ConstrainedDecodingPipeline(), default_results_path="results/cd_results.json")
"""

from __future__ import annotations

import dataclasses
import os
import sys
from collections import Counter
from typing import TYPE_CHECKING

from experiments.shared_pipeline_elements.aci_data_loader import load_test1_encounters, save_predictions
from pipeline_base import PipelineResult, SOAPPipeline, soap_note_to_text
from experiments.shared_pipeline_elements.pydantic_schema import SOAPNote

if TYPE_CHECKING:
    pass

# ─────────────────────────────────────────────
# DEFAULT PATHS
# ─────────────────────────────────────────────
DATA_CSV     = "data/clinicalnlp_taskB_test1.csv"
METADATA_CSV = "data/clinicalnlp_taskB_test1_metadata.csv"
MAX_RETRIES  = 2   # only meaningful for GVR; others use 0


# ─────────────────────────────────────────────
# RESULT SERIALIZATION
# ─────────────────────────────────────────────
def save_results(results: list[PipelineResult], path: str) -> None:
    """Serialize full pipeline results to JSON (incremental save after each record)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, indent=2)


def load_results(path: str) -> list[PipelineResult]:
    """Reload previously saved results for resumable batch runs."""
    with open(path) as f:
        return [PipelineResult(**r) for r in json.load(f)]


def results_to_predictions(results: list[PipelineResult]) -> dict[str, str]:
    """
    Convert pipeline results to {encounter_id: note_text} for evaluation.
    Failed records get an empty string so evaluation still runs on all 40.
    """
    predictions = {}
    for r in results:
        if r.success and r.soap_note:
            try:
                soap = SOAPNote(**r.soap_note)
                predictions[r.encounter_id] = soap_note_to_text(soap, r.encounter_id)
            except Exception:
                predictions[r.encounter_id] = ""
        else:
            predictions[r.encounter_id] = ""
    return predictions


# ─────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────
def run_batch(
    pipeline: SOAPPipeline,
    encounters: list,
    max_retries: int = MAX_RETRIES,
    save_path: str = "results/results.json",
) -> list[PipelineResult]:
    """
    Run a pipeline over a list of encounters with resumable checkpointing.

    After each record, results are written to save_path and a predictions
    CSV is written alongside it. If the job is interrupted and rerun,
    already-completed encounters are skipped automatically.

    Args:
        pipeline:    Any SOAPPipeline subclass instance.
        encounters:  List of ACIEncounter from load_test1_encounters().
        max_retries: Passed through to pipeline.run_pipeline(). GVR uses 2,
                     prompt-eng and constrained decoding should use 0.
        save_path:   Where to write results JSON.

    Returns:
        List of PipelineResult (all encounters, including pre-existing).
    """
    results: list[PipelineResult] = []
    completed_ids: set[str] = set()

    # Resume from existing file if present
    if os.path.exists(save_path):
        results = load_results(save_path)
        completed_ids = {r.encounter_id for r in results}
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} encounters already done in {save_path}")

    for i, enc in enumerate(encounters):
        if enc.encounter_id in completed_ids:
            print(f"[{i+1}/{len(encounters)}] Skipping {enc.encounter_id} (already done)")
            continue

        print(f"\n[{i+1}/{len(encounters)}] Processing {enc.encounter_id} ...")
        result = pipeline.run_pipeline(enc, max_retries=max_retries)
        results.append(result)

        status = "OK" if result.success else "FAIL"
        print(f"  [{status}] attempts={result.attempts} | "
              f"failed_fields={result.failed_fields} | "
              f"plan_inferred={result.plan_is_inferred}")

        # Incremental save after every record
        save_results(results, save_path)

        predictions = results_to_predictions(results)
        pred_path = save_path.replace(".json", "_predictions.csv")
        save_predictions(predictions, pred_path, encounters=encounters)

    return results


# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────
def print_summary(results: list[PipelineResult]) -> None:
    """Print a pass/fail/retry breakdown to stdout."""
    total      = len(results)
    successes  = [r for r in results if r.success]
    failures   = [r for r in results if not r.success]
    first_pass = [r for r in successes if r.attempts == 1]
    retried    = [r for r in successes if r.attempts > 1]
    inferred   = [r for r in successes if r.plan_is_inferred]

    techniques = Counter(r.technique for r in results)

    print(f"\n{'='*52}")
    print(f"PIPELINE SUMMARY  ({total} encounters)")
    if len(techniques) > 1:
        print(f"Techniques: {dict(techniques)}")
    print(f"{'='*52}")
    print(f"Overall pass rate:     {len(successes)}/{total} ({100*len(successes)/total:.1f}%)")
    print(f"First-pass rate:       {len(first_pass)}/{total} ({100*len(first_pass)/total:.1f}%)")
    print(f"Succeeded after retry: {len(retried)}/{total} ({100*len(retried)/total:.1f}%)")
    print(f"Failed after retries:  {len(failures)}/{total} ({100*len(failures)/total:.1f}%)")
    print(f"Plan inferred:         {len(inferred)}/{len(successes)} successes")

    dist = Counter(r.attempts for r in results)
    print(f"\nAttempt distribution:")
    for k in sorted(dist):
        print(f"  {k} attempt(s): {dist[k]} records")

    if failures:
        ftypes = Counter(r.failure_type for r in failures)
        print(f"\nFailure types (unresolved after retries):")
        for k, v in ftypes.most_common():
            print(f"  {k}: {v}")

    all_fields = [f for r in results for f in r.failed_fields]
    if all_fields:
        print(f"\nMost common failed fields (across all attempts):")
        for f, count in Counter(all_fields).most_common(5):
            print(f"  {f}: {count}")

    print(f"{'='*52}\n")


# ─────────────────────────────────────────────
# SHARED CLI ENTRY POINT
# ─────────────────────────────────────────────
def main(
    pipeline: SOAPPipeline,
    default_results_path: str = "results/results.json",
    default_max_retries: int = 0,
) -> None:
    """
    Standard CLI used by all three technique scripts.

    Usage:
        python <technique>.py single
        python <technique>.py batch
        python <technique>.py batch --n 5

    Args:
        pipeline:             An instantiated SOAPPipeline subclass.
        default_results_path: Where to save results (technique scripts pass their own path).
        default_max_retries:  GVR passes 2; others pass 0.
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    n = None
    if "--n" in sys.argv:
        try:
            n = int(sys.argv[sys.argv.index("--n") + 1])
        except (IndexError, ValueError):
            print(f"Usage: python <technique>.py [single|batch] [--n N]")
            sys.exit(1)

    encounters = load_test1_encounters(DATA_CSV, METADATA_CSV)

    if mode == "single":
        enc = encounters[0]
        print(f"Running single record: {enc.encounter_id}")
        result = pipeline.run_pipeline(enc, max_retries=default_max_retries)
        status = "Success" if result.success else "Failed"
        print(f"\n[{status}] attempts={result.attempts}")
        if result.success:
            print(json.dumps(result.soap_note, indent=2))
            print("\n--- Generated note text ---")
            soap = SOAPNote(**result.soap_note)
            print(soap_note_to_text(soap, result.encounter_id))
        else:
            print(f"Error: {result.final_error}")

    elif mode == "batch":
        if n is not None:
            encounters = encounters[:n]
        save_path = (
            default_results_path.replace(".json", f"_n{len(encounters)}.json")
            if n else default_results_path
        )
        print(f"Running batch on {len(encounters)} encounters "
              f"[technique={pipeline.TECHNIQUE_NAME}] ...")
        results = run_batch(
            pipeline,
            encounters,
            max_retries=default_max_retries,
            save_path=save_path,
        )
        print_summary(results)
        print(f"Results:     {save_path}")
        print(f"Predictions: {save_path.replace('.json', '_predictions.csv')}")

    else:
        print(f"Usage: python <technique>.py [single|batch] [--n N]")
        print(f"  single      — run on first encounter only (quick test)")
        print(f"  batch       — run on all 40 encounters")
        print(f"  batch --n 5 — run on first 5 encounters only")


# allow `import json` at the top of main() without a local import
import json
