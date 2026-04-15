"""
generate_validate_retry.py
──────────────────────────
GVR technique: generate → validate (JSON + Pydantic) → retry with feedback.

This file contains only GVR-specific logic:
  - JSON repair parser
  - The retry loop with feedback injection

Run:
    PYTHONPATH=. python experiments/generate_validate_retry.py single
    PYTHONPATH=. python experiments/generate_validate_retry.py -- n 5
    PYTHONPATH=. python experiments/generate_validate_retry.py batch

Everything shared (LLM calls, PipelineResult, batch runner, text formatting)
lives in pipeline_base.py and batch_runner.py.
"""

from __future__ import annotations

import json
from pydantic import ValidationError

from experiments.shared_pipeline_elements.aci_data_loader import ACIEncounter
from gvr_prompt_templates import build_initial_prompt, build_retry_prompt, format_validation_error
from experiments.shared_pipeline_elements.pydantic_schema import SOAPNote
from experiments.shared_pipeline_elements.pipeline_base import SOAPPipeline, PipelineResult

MAX_RETRIES = 2  # max 2 retries = 3 total LLM calls per record (Dao et al.)


# ─────────────────────────────────────────────
# GVR-SPECIFIC HELPERS
# ─────────────────────────────────────────────
def parse_json_response(raw: str) -> dict:
    """
    Strip markdown fences if present, attempt repair if needed, then parse JSON.
    This is GVR-specific because the other techniques don't do free-form JSON parsing.
    """
    clean = raw.strip()

    # strip markdown fences
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1]).strip()

    # attempt direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # repair: find outermost { ... } and try again
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(clean[start:end+1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse JSON from response", clean, 0)


# ─────────────────────────────────────────────
# GVR PIPELINE
# ─────────────────────────────────────────────
class GVRPipeline(SOAPPipeline):
    """
    Generate-Validate-Retry pipeline.

    Loop (up to max_retries + 1 times):
      1. Call LLM with the current prompt
      2. Parse JSON (repair if needed) — retry with error feedback on failure
      3. Validate with Pydantic — retry with field-level feedback on failure
      4. Return success on first clean validation
    """

    TECHNIQUE_NAME = "gvr"

    def run_pipeline(
        self,
        encounter: ACIEncounter,
        max_retries: int = MAX_RETRIES,
    ) -> PipelineResult:

        prompt            = build_initial_prompt(encounter.transcript, encounter.encounter_id)
        last_error        = None
        failure_type      = None
        all_failed_fields = []

        for attempt in range(max_retries + 1):
            try:
                raw = self.call_llm(prompt)

                # ── Layer 1: JSON parse ───────────────────────────────
                try:
                    parsed = parse_json_response(raw)
                except json.JSONDecodeError as e:
                    failure_type = "json_parse"
                    last_error   = f"Invalid JSON: {e}"
                    print(f"  [{encounter.encounter_id}] Attempt {attempt+1} — JSON parse error")
                    prompt = build_retry_prompt(
                        encounter.transcript, encounter.encounter_id, raw, last_error
                    )
                    continue

                # ── Layer 2: Pydantic validation ──────────────────────
                try:
                    # inject scenario_id placeholder if schema still requires it
                    if "scenario_id" in SOAPNote.model_fields and "scenario_id" not in parsed:
                        parsed["scenario_id"] = 0

                    soap = SOAPNote(**parsed)
                    return self._success(
                        encounter,
                        soap,
                        attempts=attempt + 1,
                        failed_fields=all_failed_fields,
                    )

                except ValidationError as e:
                    failure_type  = "pydantic"
                    failed_fields = list({err["loc"][0] for err in e.errors() if err["loc"]})
                    all_failed_fields.extend(failed_fields)
                    last_error    = format_validation_error(e)
                    print(f"  [{encounter.encounter_id}] Attempt {attempt+1} "
                          f"— Pydantic error: {failed_fields}")
                    prompt = build_retry_prompt(
                        encounter.transcript, encounter.encounter_id, raw, last_error
                    )
                    continue

            except Exception as e:
                failure_type = "unexpected"
                last_error   = f"Unexpected error: {e}"
                print(f"  [{encounter.encounter_id}] Attempt {attempt+1} — Unexpected: {e}")
                break

        # all retries exhausted
        return self._failure(
            encounter,
            error=last_error,
            failure_type=failure_type,
            attempts=max_retries + 1,
            failed_fields=all_failed_fields,
        )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from experiments.shared_pipeline_elements.batch_runner import main
    main(
        GVRPipeline(),
        default_results_path="../results/constrained_JSON_output/gvr_results.json",
        default_max_retries=MAX_RETRIES,
    )