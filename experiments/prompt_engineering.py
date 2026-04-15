"""
prompt_engineering.py
─────────────────────
Prompt engineering technique: carefully crafted prompt, parse output,
validate with Pydantic. No retry loop — prompt quality is the structural guarantee.

Only technique-specific logic lives here. Everything shared
(LLM calls, PipelineResult, batch runner) is in pipeline_base.py / batch_runner.py.

Run:
    PYTHONPATH=. python experiments/prompt_engineering.py single
    PYTHONPATH=. python experiments/prompt_engineering.py -- n 5
    PYTHONPATH=. python experiments/prompt_engineering.py batch
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from shared_pipeline_elements.aci_data_loader import ACIEncounter
from generate_validate_retry import parse_json_response              # reuse GVR's robust JSON repair parser
from gvr_prompt_templates import COMPACT_SCHEMA  # stays in sync with SOAPNote automatically
from shared_pipeline_elements.pydantic_schema import SOAPNote
from shared_pipeline_elements.pipeline_base import SOAPPipeline, PipelineResult



# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def _build_prompt(encounter: ACIEncounter) -> str:
    """
    A single structured prompt designed to produce valid SOAPNote JSON
    without any retry. Three techniques are combined:

    1. Role + task framing   — anchors the model in clinical documentation context
    2. Extraction guide      — explicit section-by-section mapping rules replace
                               the GVR working_space scratchpad; reasoning is done
                               inline so the model doesn't need a correction pass
    3. Output contract       — strict rules for null vs empty string, boolean
                               handling, and typed field extraction, with examples
    """

    # Pull metadata if available — gives the model authoritative anchors
    # so it doesn't have to infer them from transcript language alone
    meta_block = ""
    if encounter.chief_complaint:
        meta_block += f"Chief complaint (authoritative): {encounter.chief_complaint}\n"
    if encounter.patient_age:
        meta_block += f"Patient age: {encounter.patient_age}\n"
    if encounter.patient_gender:
        meta_block += f"Patient gender: {encounter.patient_gender}\n"
    if meta_block:
        meta_block = f"\nCLINICAL METADATA (use these as ground truth — do not override from transcript):\n{meta_block}"

    return f"""You are an experienced clinical documentation specialist. Your task is to read a doctor-patient conversation transcript and produce a structured SOAP note as a single valid JSON object.

══════════════════════════════════════════
EXTRACTION GUIDE — work through each step mentally before writing JSON:

Step 1 — Read the full transcript start to finish. Do not stop early.

Step 2 — Map content to the four required text divisions:
  • subjective          → patient's verbal report: HPI, PMH, medications, allergies, social/family history
  • objective_exam      → what the clinician physically observed or measured: vitals, auscultation, palpation, neuro
  • objective_results   → lab values, imaging, diagnostic test results discussed in the visit
                          USE EMPTY STRING "" if no labs or imaging are mentioned — never null
  • assessment_and_plan → clinician's diagnosis/impression + next steps: treatments, orders, referrals, follow-up

Step 3 — Collect typed values ONLY if explicitly stated:
  • pain_severity:  only if a 0–10 numeric score is explicitly given — not "severe", not "a lot of pain"
  • temperature_f:  Fahrenheit only — convert from Celsius if needed (°C × 9/5 + 32)
  • bp_systolic / bp_diastolic: split from "120/80" → systolic=120, diastolic=80
  • All other numeric fields: extract only if a number is explicitly stated

Step 4 — Boolean fields (smoker, alcohol_use, recreational_drugs, is_urgent):
  • true  → explicitly confirmed in transcript ("yes I smoke", "patient drinks daily")
  • false → explicitly denied ("denies smoking", "no alcohol use")
  • null  → not mentioned at all

Step 5 — Null vs empty string rule (CRITICAL):
  • Optional typed fields not mentioned → JSON null   (e.g. "pain_severity": null)
  • objective_results with no labs/imaging → ""       (e.g. "objective_results": "")
  • NEVER use the string "null" or "N/A" — use actual JSON null

══════════════════════════════════════════
OUTPUT CONTRACT:
  • Output exactly ONE JSON object — no preamble, no explanation, no markdown code fences
  • ALL fields must be FLAT top-level keys — do NOT nest objects. Wrong:
      {{"patient": {{"age": 45, "gender": "male"}}}}
    Correct:
      {{"patient_age": 45, "patient_gender": "male"}}
  • String fields must be plain strings — do NOT use objects. Wrong:
      {{"allergies": {{"seasonal": "none"}}}}
    Correct:
      {{"allergies": "No known allergies"}}
  • All five required text fields (chief_complaint, subjective, objective_exam,
    objective_results, assessment_and_plan) must be present as top-level keys
    with string values — objective_results may be ""
  • Do not fabricate clinical information not present in the transcript
  • Do not repeat any JSON keys
{meta_block}
══════════════════════════════════════════
SCHEMA:
{COMPACT_SCHEMA}

══════════════════════════════════════════
ENCOUNTER ID: {encounter.encounter_id}

TRANSCRIPT:
{encounter.transcript}

══════════════════════════════════════════
PRE-FLIGHT CHECK — verify mentally before writing JSON:
  □ Have I read to the end of the transcript (plans are often stated last)?
  □ Are all five required text fields non-empty (except objective_results)?
  □ Is objective_results "" rather than null if no labs/imaging were discussed?
  □ Are boolean fields null (not false) when the topic was never mentioned?
  □ Are numeric typed fields null when no explicit number was stated?
  □ Is encounter_id copied exactly as shown above?

JSON:"""


# ─────────────────────────────────────────────
# PROMPT ENGINEERING PIPELINE
# ─────────────────────────────────────────────

class PromptEngineeringPipeline(SOAPPipeline):
    """
    Single-shot prompt engineering technique.

    The prompt is structured to eliminate the failure modes that GVR's retry
    loop corrects for:
      - Null vs empty string confusion          → explicit Step 5 rule
      - Boolean false vs null confusion         → explicit Step 4 rule
      - Missing required text fields            → pre-flight checklist
      - Numeric fields inferred from adjectives → explicit Step 3 rule
      - Truncated transcript reading            → Step 1 + pre-flight reminder

    Shares parse_json_response() with GVR for JSON repair robustness.
    Validates with Pydantic identically to GVR — failures are recorded but
    not retried (single-shot by design).
    """

    TECHNIQUE_NAME = "prompt_engineering"

    def run_pipeline(
        self,
        encounter: ACIEncounter,
        max_retries: int = 0,   # single-shot by design — prompt is the guarantee
    ) -> PipelineResult:

        try:
            raw    = self.call_llm(_build_prompt(encounter))
            parsed = parse_json_response(raw)   # GVR's repair parser handles fences + brace extraction

            # inject scenario_id placeholder if schema still requires it
            if "scenario_id" in SOAPNote.model_fields and "scenario_id" not in parsed:
                parsed["scenario_id"] = 0

            soap = SOAPNote(**parsed)
            return self._success(encounter, soap, attempts=1)

        except json.JSONDecodeError as e:
            return self._failure(
                encounter,
                error=f"Invalid JSON: {e}",
                failure_type="json_parse",
                attempts=1,
            )
        except ValidationError as e:
            failed_fields = list({err["loc"][0] for err in e.errors() if err["loc"]})
            return self._failure(
                encounter,
                error=str(e),
                failure_type="pydantic",
                attempts=1,
                failed_fields=failed_fields,
            )
        except Exception as e:
            return self._failure(
                encounter,
                error=str(e),
                failure_type="unexpected",
                attempts=1,
            )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from shared_pipeline_elements.batch_runner import main
    main(
        PromptEngineeringPipeline(),
        default_results_path="results/pe_results.json",
        default_max_retries=0,  # single-shot by design
    )