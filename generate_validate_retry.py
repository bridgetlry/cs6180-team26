import json
import time
import requests
import os
from dataclasses import dataclass, field as dc_field
from typing import Optional
from pydantic import ValidationError
from dotenv import load_dotenv

from aci_data_loader import ACIEncounter, load_train, load_test, format_transcript
from soap_schema import SOAPNote
from prompt_templates import build_initial_prompt, build_retry_prompt, format_validation_error

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL              = "meta-llama/llama-3.1-8b-instruct"
API_URL            = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES        = 2       # max 3 total LLM calls per record (Dao et al.)
DELAY_SECONDS      = 0.5     # between calls — increase if rate limited
TEMPERATURE        = 0.0     # deterministic for reproducibility
RESULTS_PATH       = "gvr_results.json"


# ─────────────────────────────────────────────
# RESULT DATACLASS
# Captures everything needed for evaluation:
# structural metrics, retry behavior, failure taxonomy
# ─────────────────────────────────────────────
@dataclass
class PipelineResult:
    # identity
    encounter_id: str
    dataset_subset: str           # virtassist | virtscribe | aci

    # outcome
    success: bool
    attempts: int                 # 1, 2, or 3
    soap_note: Optional[dict]     # validated SOAPNote as dict, None if failed

    # failure tracking (for failure taxonomy)
    final_error: Optional[str]
    failure_type: Optional[str]   # json_parse | pydantic | unexpected | None
    failed_fields: list[str] = dc_field(default_factory=list)

    # semantic flags
    plan_is_inferred: Optional[bool] = None

    # ground truth from metadata (for evaluation)
    chief_complaint_gt: Optional[str] = None   # from metadata CSV
    patient_age_gt: Optional[int] = None        # from metadata CSV
    patient_gender_gt: Optional[str] = None     # from metadata CSV

    # reference note (for LLM-as-judge evaluation)
    reference_note: Optional[str] = None


# ─────────────────────────────────────────────
# LLM WRAPPER
# ─────────────────────────────────────────────
def call_llm(prompt: str, temperature: float = TEMPERATURE) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 3000
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def parse_json_response(raw: str) -> dict:
    """
    Strip markdown fences if present, attempt repair if needed, then parse JSON.
    Handles common LLM formatting issues:
    - Missing opening brace
    - Markdown code fences
    - Leading/trailing whitespace
    """
    clean = raw.strip()

    # strip markdown fences
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1]).strip()

    # attempt direct parse first
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # repair: find first { and last } and extract that substring
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(clean[start:end+1])
        except json.JSONDecodeError:
            pass

    # if all repair attempts fail, raise to trigger retry
    raise json.JSONDecodeError("Could not parse JSON from response", clean, 0)


# ─────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(
    encounter: ACIEncounter,
    max_retries: int = MAX_RETRIES
) -> PipelineResult:
    """
    Generate-validate-retry pipeline for a single ACIEncounter.

    Architecture mirrors Dao et al. (2025):
    - JSON validation layer
    - Pydantic schema enforcement
    - Retry with targeted error feedback
    - Maximum two retries (three total LLM calls)
    """
    transcript   = format_transcript(encounter)
    encounter_id = encounter.encounter_id
    prompt       = build_initial_prompt(transcript, encounter_id)

    last_error      = None
    failure_type    = None
    all_failed_fields = []

    for attempt in range(max_retries + 1):
        try:
            raw = call_llm(prompt)
            time.sleep(DELAY_SECONDS)

            # ── Layer 1: JSON validation ─────────────────────────────
            try:
                parsed = parse_json_response(raw)
            except json.JSONDecodeError as e:
                failure_type = "json_parse"
                last_error   = f"Invalid JSON: {str(e)}"
                print(f"  [{encounter_id}] Attempt {attempt+1} — JSON parse error")
                prompt = build_retry_prompt(
                    transcript, encounter_id, raw, last_error
                )
                continue

            # ── Layer 2: Pydantic schema enforcement ─────────────────
            try:
                soap = SOAPNote(**parsed)

                # success
                return PipelineResult(
                    encounter_id      = encounter_id,
                    dataset_subset    = encounter.dataset,
                    success           = True,
                    attempts          = attempt + 1,
                    soap_note         = soap.model_dump(),
                    final_error       = None,
                    failure_type      = None,
                    failed_fields     = all_failed_fields,
                    plan_is_inferred  = soap.plan_is_inferred,
                    chief_complaint_gt= encounter.chief_complaint_gt,
                    patient_age_gt    = None,  # TODO: add to ACIEncounter from metadata
                    patient_gender_gt = encounter.gender,
                    reference_note    = encounter.reference_note,
                )

            except ValidationError as e:
                failure_type  = "pydantic"
                failed_fields = list({err["loc"][0] for err in e.errors() if err["loc"]})
                all_failed_fields.extend(failed_fields)
                last_error    = format_validation_error(e)
                print(f"  [{encounter_id}] Attempt {attempt+1} — Pydantic error: {failed_fields}")
                prompt = build_retry_prompt(
                    transcript, encounter_id, raw, last_error
                )
                continue

        except Exception as e:
            failure_type = "unexpected"
            last_error   = f"Unexpected error: {str(e)}"
            print(f"  [{encounter_id}] Attempt {attempt+1} — Unexpected: {e}")
            break

    # all retries exhausted
    return PipelineResult(
        encounter_id      = encounter_id,
        dataset_subset    = encounter.dataset,
        success           = False,
        attempts          = max_retries + 1,
        soap_note         = None,
        final_error       = last_error,
        failure_type      = failure_type,
        failed_fields     = all_failed_fields,
        plan_is_inferred  = None,
        chief_complaint_gt= encounter.chief_complaint_gt,
        patient_age_gt    = None,
        patient_gender_gt = encounter.gender,
        reference_note    = encounter.reference_note,
    )


def _safe_int(val) -> Optional[int]:
    """Safely convert metadata age to int."""
    try:
        return int(float(val)) if val else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────
def run_batch(
    encounters: list[ACIEncounter],
    max_retries: int = MAX_RETRIES,
    save_path: str = RESULTS_PATH
) -> list[PipelineResult]:
    """
    Run pipeline on a list of encounters.
    Saves results incrementally to JSON after each record
    so progress is not lost if the run is interrupted.
    """
    results = []

    for i, enc in enumerate(encounters):
        print(f"\n[{i+1}/{len(encounters)}] Processing {enc.encounter_id}...")
        result = run_pipeline(enc, max_retries=max_retries)
        results.append(result)

        status = "✅" if result.success else "❌"
        print(f"  {status} attempts={result.attempts} | "
              f"failed_fields={result.failed_fields} | "
              f"plan_inferred={result.plan_is_inferred}")

        # save incrementally
        _save_results(results, save_path)

    return results


def _save_results(results: list[PipelineResult], path: str):
    """Serialize results to JSON."""
    import dataclasses
    with open(path, "w") as f:
        json.dump(
            [dataclasses.asdict(r) for r in results],
            f, indent=2
        )


# ─────────────────────────────────────────────
# SUMMARY STATS
# Quick structural metrics after a batch run
# ─────────────────────────────────────────────
def print_summary(results: list[PipelineResult]):
    total       = len(results)
    successes   = [r for r in results if r.success]
    failures    = [r for r in results if not r.success]
    first_pass  = [r for r in successes if r.attempts == 1]
    retried     = [r for r in successes if r.attempts > 1]
    inferred    = [r for r in successes if r.plan_is_inferred]

    print(f"\n{'='*50}")
    print(f"PIPELINE SUMMARY — {total} encounters")
    print(f"{'='*50}")
    print(f"Overall pass rate:    {len(successes)}/{total} ({100*len(successes)/total:.1f}%)")
    print(f"First-pass rate:      {len(first_pass)}/{total} ({100*len(first_pass)/total:.1f}%)")
    print(f"Succeeded after retry:{len(retried)}/{total} ({100*len(retried)/total:.1f}%)")
    print(f"Failed after retries: {len(failures)}/{total} ({100*len(failures)/total:.1f}%)")
    print(f"Plan inferred:        {len(inferred)}/{len(successes)} ({100*len(inferred)/max(len(successes),1):.1f}%)")

    # attempt distribution
    from collections import Counter
    dist = Counter(r.attempts for r in results)
    print(f"\nAttempt distribution:")
    for k in sorted(dist):
        print(f"  {k} attempt(s): {dist[k]} records")

    # failure type breakdown
    if failures:
        ftypes = Counter(r.failure_type for r in failures)
        print(f"\nFailure types (unresolved after retries):")
        for k, v in ftypes.most_common():
            print(f"  {k}: {v}")

    # most commonly failed fields
    from collections import Counter as C
    all_fields = []
    for r in results:
        all_fields.extend(r.failed_fields)
    if all_fields:
        print(f"\nMost common failed fields (across all attempts):")
        for field, count in C(all_fields).most_common(5):
            print(f"  {field}: {count}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    if mode == "train":
        print("Running on full train split...")
        encounters = load_train()
        results = run_batch(encounters, save_path="gvr_train_results.json")
        print_summary(results)

    elif mode == "test":
        print("Running on test split...")
        encounters = load_test()
        results = run_batch(encounters, save_path="gvr_test_results.json")
        print_summary(results)

    elif mode == "single":
        # run on first training record for debugging
        encounters = load_train()
        enc = encounters[0]
        print(f"Running single record: {enc.encounter_id}")
        result = run_pipeline(enc)
        status = "✅ Success" if result.success else "❌ Failed"
        print(f"\n{status} | attempts={result.attempts}")
        if result.success:
            print(json.dumps(result.soap_note, indent=2))
        else:
            print(f"Error: {result.final_error}")

    else:
        print("Usage: python generate_validate_retry.py [train|test|single]")