"""
llm_judge.py
────────────
LLM-as-judge for typed field extraction quality across GVR, PE, and CD pipelines.

The judge evaluates ONLY the typed/structured fields in the SOAPNote output —
not the narrative SOAP sections (those are evaluated by ROUGE and MEDCON).

Taxonomy categories (Dao et al. 2025 — exact names):

  STRUCTURAL (automated — no judge needed):
    json_parse       — model returned non-JSON or unparseable output
    pydantic         — model returned JSON but failed Pydantic field validation
    pipeline_error   — unexpected exception during processing

  TYPED-FIELD SEMANTIC (LLM judge — mirrors Dao et al. Figure 4):
    hallucinated     — typed field contains a fabricated value not in the transcript
    missed           — typed field is null but value is explicitly stated in transcript
    hallucinated_and_missed — both present simultaneously (Dao et al. "Missed & Misplaced")
    accurate         — all typed fields correctly extracted or legitimately null

  AUTOMATED:
    instruction_bleed — prompt artifacts detected in narrative output (string match)

Priority for taxonomy_label:
  json_parse > pydantic > pipeline_error > hallucinated >
  missed > instruction_bleed > accurate

Run from project root:
    # Judge a single technique's results:
    python evaluation/llm_judge.py \
        --results results/constrained_JSON_output/gvr_results.json \
        --transcripts data/clinicalnlp_taskB_test1.csv

    # Judge all three techniques and compare:
    python evaluation/llm_judge.py \
        --results results/constrained_JSON_output/gvr_results.json \
                  results/constrained_JSON_output/pe_results.json \
                  results/constrained_JSON_output/cd_results.json \
        --transcripts data/clinicalnlp_taskB_test1.csv \
        --output judge_results.json

    # Limit to first N records per technique (for testing):
    python evaluation/llm_judge.py \
        --results results/constrained_JSON_output/gvr_results.json \
        --transcripts data/clinicalnlp_taskB_test1.csv \
        --n 5

Requirements:
    pip install requests pandas python-dotenv
    Set OPENROUTER_API_KEY in your .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import requests
from dataclasses import dataclass, asdict, field
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
JUDGE_MODEL   = "anthropic/claude-sonnet-4-5"
API_URL       = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS    = 2000
DELAY_SECONDS = 0.5

# Patterns for automated instruction_bleed detection (case-insensitive)
INSTRUCTION_BLEED_PATTERNS = [
    "here is your soap note",
    "here is the soap note",
    "here's the soap note",
    "i hope this helps",
    "please note that",
    "as an ai",
    "as a language model",
    "i cannot provide",
    "i am unable to",
    "```json",
    "```",
    "json:",
    "note: this",
    "disclaimer:",
]


# ─────────────────────────────────────────────
# JUDGE PROMPTS
# ─────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an expert clinical documentation auditor and AI evaluation specialist.

Your task is to assess ONLY the typed/structured fields extracted from a doctor-patient transcript.
Do NOT evaluate the narrative SOAP sections (chief_complaint, subjective, objective_exam,
objective_results, assessment_and_plan) — those are evaluated separately by other metrics.

TYPED FIELD FAILURE TAXONOMY — Dao et al. 2025 exact categories:

  hallucinated  — a typed field contains a value that is NOT explicitly stated in the
                  transcript. This is the most clinically dangerous failure.

                  Examples:
                    • bp_systolic: 120 when the doctor only said "vitals look great"
                      with no numbers
                    • pain_severity: 8.0 when no numeric pain score was stated
                    • smoker: true when smoking was never mentioned

                  Be conservative: only flag if you are confident the value is absent
                  from the transcript. Format each example as:
                    "field_name: value X — transcript never stated this value"

  missed        — a typed field is null in the output BUT its value is explicitly
                  and unambiguously stated in the transcript.

                  Typed fields to check:
                    patient_age, patient_gender, symptom_duration,
                    pain_severity, pain_quality, pain_location,
                    smoker, alcohol_use, recreational_drugs,
                    bp_systolic, bp_diastolic, heart_rate, o2_saturation,
                    respiratory_rate, temperature_f,
                    hemoglobin_a1c, blood_glucose, systolic_murmur_grade,
                    muscle_strength_right, muscle_strength_left,
                    kidney_stone_size_mm, bladder_volume_prevoid_ml,
                    bladder_volume_postvoid_ml, wound_length_cm, wound_width_cm,
                    bsa, medication_count.

                  Flag ONLY if ALL three are true:
                    1. The value is explicitly stated as a number, boolean, or
                       specific term in the transcript (not implied or inferred)
                    2. The corresponding field is null in the TYPED FIELDS output
                    3. The value is unambiguous — e.g. "pain is 7 out of 10",
                       "blood pressure 120 over 80", "he does smoke"

                  Do NOT flag:
                    • Ambiguous or implied values
                    • Values present only in the reference note, not the transcript
                    • Fields that are legitimately null (value not mentioned)

                  Format each example as:
                    "field_name: transcript says X but field is null"

IMPORTANT NOTES:
  • Evaluate TYPED FIELDS ONLY. Ignore narrative section content entirely.
  • The reference note is provided for context only — ground all assessments in
    the transcript.
  • json_parse and pydantic failures are structural and detected automatically —
    do not assess them.
  • If TYPED FIELDS shows "(no structured output — structural failure)", set all
    flags to false.

Respond ONLY with valid JSON — no preamble, no markdown, no explanation outside the JSON."""


JUDGE_USER_TEMPLATE = """Assess the typed/structured field extractions for this clinical encounter.
Evaluate ONLY the TYPED FIELDS — do not assess the narrative SOAP sections.

ENCOUNTER ID: {encounter_id}
TECHNIQUE: {technique}

SOURCE TRANSCRIPT:
{transcript}

TYPED FIELDS (extracted structured values — this is what you are evaluating):
{typed_fields}

Return a JSON object with exactly this structure:
{{
  "encounter_id": "{encounter_id}",
  "technique": "{technique}",
  "hallucinated": true/false,
  "missed": true/false,
  "hallucinated_examples": ["field_name: value X — transcript never stated this value", ...],
  "missed_examples": ["field_name: transcript says X but field is null", ...],
  "overall_failure_type": "hallucinated" | "missed" | "hallucinated_and_missed" | "accurate",
  "severity": "none" | "minor" | "moderate" | "severe",
  "reasoning": "2-4 sentence explanation focused on typed field extraction quality"
}}

Severity guide (typed fields only):
  none     — all extractable typed fields are correct or legitimately null (accurate)
  minor    — small extraction error that does not affect clinical meaning
  moderate — extraction error that could affect clinical interpretation
  severe   — fabricated or missed value that could cause direct patient harm
             (e.g. wrong pain score, fabricated vital signs, missed drug allergy flag)

JSON:"""


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class JudgeResult:
    encounter_id: str
    technique: str

    # Structural failures (from pipeline results — no LLM judge needed)
    structural_failure_type: Optional[str]    # json_parse | pydantic | pipeline_error | None
    structural_failed_fields: list
    attempts: int

    # Automated pre-check
    instruction_bleed: bool
    instruction_bleed_examples: list

    # Typed-field failures (from LLM judge — Dao et al. 2025 exact category names)
    hallucinated: bool
    missed: bool
    hallucinated_examples: list
    missed_examples: list
    overall_failure_type: str    # hallucinated | missed | hallucinated_and_missed | accurate
    severity: str
    reasoning: str

    # Derived — "accurate" mirrors Dao et al. Figure 4 label
    taxonomy_label: str = "accurate"   # accurate | hallucinated | missed | hallucinated_and_missed
                                       # | instruction_bleed | json_parse | pydantic | pipeline_error
    judge_skipped: bool = False               # True if judge was not called (structural failure)
    judge_raw: Optional[dict] = None          # raw judge response for debugging


def compute_taxonomy_label(result: JudgeResult) -> str:
    """
    Assign the single most specific failure type using Dao et al. 2025 exact names.
    Priority: structural > hallucinated > missed > instruction_bleed > accurate
    Co-occurring hallucinated+missed maps to Dao's "Missed & Misplaced" concept.
    """
    if result.structural_failure_type in ("json_parse", "pydantic", "pipeline_error"):
        return result.structural_failure_type
    if result.hallucinated and result.missed:
        return "hallucinated_and_missed"
    if result.hallucinated:
        return "hallucinated"
    if result.missed:
        return "missed"
    if result.instruction_bleed:
        return "instruction_bleed"
    return "accurate"


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def load_transcripts(csv_path: str) -> dict[str, str]:
    """Load encounter_id -> transcript mapping from ACI-Bench CSV."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["encounter_id"], df["dialogue"]))


def load_pipeline_results(json_path: str) -> list[dict]:
    """Load pipeline results from a technique's JSON output."""
    with open(json_path) as f:
        return json.load(f)


def note_from_result(result: dict) -> Optional[str]:
    """
    Reconstruct the plain-text note from a pipeline result dict.
    Returns None if the result failed structurally.
    """
    soap = result.get("soap_note")
    if not soap:
        return None

    parts = []
    if soap.get("chief_complaint"):
        parts.append(f"CHIEF COMPLAINT\n\n{soap['chief_complaint']}")
    if soap.get("subjective"):
        parts.append(f"HISTORY OF PRESENT ILLNESS\n\n{soap['subjective']}")
    if soap.get("objective_exam"):
        parts.append(f"PHYSICAL EXAM\n\n{soap['objective_exam']}")
    if soap.get("objective_results") and soap["objective_results"].strip():
        parts.append(f"RESULTS\n\n{soap['objective_results']}")
    if soap.get("assessment_and_plan"):
        parts.append(f"ASSESSMENT AND PLAN\n\n{soap['assessment_and_plan']}")

    return "\n\n".join(parts) if parts else None


# Typed fields in SOAPNote schema — judge checks these for "missed" and "hallucinated"
TYPED_FIELD_NAMES = [
    "patient_age", "patient_gender", "symptom_duration",
    "pain_severity", "pain_quality", "pain_location",
    "smoker", "alcohol_use", "recreational_drugs",
    "bp_systolic", "bp_diastolic", "heart_rate", "o2_saturation",
    "respiratory_rate", "temperature_f",
    "hemoglobin_a1c", "blood_glucose",
    "systolic_murmur_grade", "wound_length_cm", "wound_width_cm",
    "kidney_stone_size_mm", "bladder_volume_prevoid_ml", "bladder_volume_postvoid_ml",
    "muscle_strength_right", "muscle_strength_left", "bsa",
    "medication_count",
]


def typed_fields_from_result(result: dict) -> str:
    """
    Extract typed fields from a pipeline result's soap_note dict and
    format them as a readable key: value list for the judge prompt.
    Shows both extracted values (non-null) and missed values (null)
    so the judge can check which explicit transcript values were missed.
    """
    soap = result.get("soap_note")
    if not soap:
        return "(no structured output — structural failure)"

    lines = []
    for field in TYPED_FIELD_NAMES:
        val = soap.get(field)
        lines.append(f"  {field}: {val if val is not None else 'null'}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# AUTOMATED PRE-CHECKS
# ─────────────────────────────────────────────
def detect_instruction_bleed(note: str) -> tuple[bool, list[str]]:
    """
    Check for prompt artifacts, preamble, or generation noise in the note.
    Returns (detected: bool, examples: list[str]).
    """
    if not note:
        return False, []

    note_lower = note.lower()
    found = [p for p in INSTRUCTION_BLEED_PATTERNS if p in note_lower]
    return bool(found), found


# ─────────────────────────────────────────────
# LLM JUDGE
# ─────────────────────────────────────────────
def call_judge_api(
    encounter_id: str,
    technique: str,
    transcript: str,
    typed_fields: str,
) -> dict:
    """Call the LLM judge via OpenRouter and return parsed JSON assessment.
    Only passes transcript and typed fields — reference note excluded to prevent
    contamination of missed field assessments.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    prompt = JUDGE_USER_TEMPLATE.format(
        encounter_id=encounter_id,
        technique=technique,
        transcript=transcript[:6000],
        typed_fields=typed_fields,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    time.sleep(DELAY_SECONDS)

    raw_text = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1]).strip()

    return json.loads(raw_text)


def _make_structural_skip_result(
    encounter_id: str,
    technique: str,
    structural_failure: str,
    structural_fields: list,
    attempts: int,
    instruction_bleed: bool,
    instruction_bleed_examples: list,
) -> JudgeResult:
    """Build a JudgeResult for structural failures — no semantic judge needed."""
    jr = JudgeResult(
        encounter_id=encounter_id,
        technique=technique,
        structural_failure_type=structural_failure,
        structural_failed_fields=structural_fields,
        attempts=attempts,
        instruction_bleed=instruction_bleed,
        instruction_bleed_examples=instruction_bleed_examples,
        hallucinated=False,
        missed=False,
        hallucinated_examples=[],
        missed_examples=[],
        overall_failure_type="accurate",
        severity="none",
        reasoning="Structural failure — typed field judge not applicable.",
        judge_skipped=True,
        judge_raw=None,
    )
    jr.taxonomy_label = compute_taxonomy_label(jr)
    return jr


def _make_error_result(
    encounter_id: str,
    technique: str,
    structural_failure: Optional[str],
    structural_fields: list,
    attempts: int,
    instruction_bleed: bool,
    instruction_bleed_examples: list,
    error: Exception,
) -> JudgeResult:
    """Build a JudgeResult when the judge API call itself fails."""
    jr = JudgeResult(
        encounter_id=encounter_id,
        technique=technique,
        structural_failure_type=structural_failure,
        structural_failed_fields=structural_fields,
        attempts=attempts,
        instruction_bleed=instruction_bleed,
        instruction_bleed_examples=instruction_bleed_examples,
        hallucinated=False,
        missed=False,
        hallucinated_examples=[],
        missed_examples=[],
        overall_failure_type="accurate",
        severity="none",
        reasoning=f"Judge API error: {error}",
        judge_skipped=False,
        judge_raw=None,
    )
    jr.taxonomy_label = compute_taxonomy_label(jr)
    return jr


def judge_single(
    pipeline_result: dict,
    transcripts: dict[str, str],
    technique_override: Optional[str] = None,
) -> JudgeResult:
    """
    Run the full judge pipeline for a single pipeline result.

    Steps:
      1. Extract metadata and reconstruct generated note
      2. Automated instruction_bleed detection
      3. Short-circuit for structural failures (no LLM call needed)
      4. LLM judge for semantic failures
      5. Build and return JudgeResult with derived taxonomy_label
    """
    encounter_id       = pipeline_result["encounter_id"]
    technique          = technique_override or pipeline_result.get("technique", "unknown")
    transcript         = transcripts.get(encounter_id, "")
    structural_failure = pipeline_result.get("failure_type")
    structural_fields  = pipeline_result.get("failed_fields", [])
    attempts           = pipeline_result.get("attempts", 1)
    typed_fields       = typed_fields_from_result(pipeline_result)

    # ── Step 1: Automated instruction_bleed check (on narrative note) ─
    generated_note = note_from_result(pipeline_result) or ""
    bleed_detected, bleed_examples = detect_instruction_bleed(generated_note)

    print(f"  [{technique}] {encounter_id} | structural={structural_failure or 'none'} "
          f"| bleed={bleed_detected}")

    # ── Step 2: Short-circuit structural failures ─────────────────────
    if structural_failure in ("json_parse", "pydantic", "pipeline_error"):
        print(f"    → Skipping judge (structural failure: {structural_failure})")
        return _make_structural_skip_result(
            encounter_id, technique, structural_failure, structural_fields,
            attempts, bleed_detected, bleed_examples,
        )

    # ── Step 3: LLM judge — typed fields only ────────────────────────
    try:
        raw = call_judge_api(
            encounter_id=encounter_id,
            technique=technique,
            transcript=transcript,
            typed_fields=typed_fields,
        )

        jr = JudgeResult(
            encounter_id=encounter_id,
            technique=technique,
            structural_failure_type=structural_failure,
            structural_failed_fields=structural_fields,
            attempts=attempts,
            instruction_bleed=bleed_detected,
            instruction_bleed_examples=bleed_examples,
            hallucinated=raw.get("hallucinated", False),
            missed=raw.get("missed", False),
            hallucinated_examples=raw.get("hallucinated_examples", []),
            missed_examples=raw.get("missed_examples", []),
            overall_failure_type=raw.get("overall_failure_type", "accurate"),
            severity=raw.get("severity", "none"),
            reasoning=raw.get("reasoning", ""),
            judge_skipped=False,
            judge_raw=raw,
        )
        jr.taxonomy_label = compute_taxonomy_label(jr)

        flags = [k for k in ("hallucinated", "missed") if getattr(jr, k)]
        print(f"    → label={jr.taxonomy_label} severity={jr.severity} flags={flags or ['accurate']}")
        return jr

    except Exception as e:
        print(f"    [ERROR] Judge API failed: {e}")
        return _make_error_result(
            encounter_id, technique, structural_failure, structural_fields,
            attempts, bleed_detected, bleed_examples, e,
        )


# ─────────────────────────────────────────────
# SUMMARY / CROSS-TECHNIQUE COMPARISON
# ─────────────────────────────────────────────
def print_summary(results: list[JudgeResult]) -> None:
    from collections import Counter

    techniques = sorted(set(r.technique for r in results))

    # Dao et al. 2025 exact category order (mirrors Figure 4)
    ORDERED_LABELS = [
        "json_parse", "pydantic", "pipeline_error",
        "hallucinated", "missed", "hallucinated_and_missed",
        "instruction_bleed", "accurate",
    ]

    print(f"\n{'='*66}")
    print(f"TYPED FIELD EXTRACTION TAXONOMY  (Dao et al. 2025)")
    print(f"{'='*66}")

    for tech in techniques:
        subset = [r for r in results if r.technique == tech]
        total  = len(subset)

        labels   = Counter(r.taxonomy_label for r in subset)
        severity = Counter(r.severity for r in subset)

        flag_counts = {
            "hallucinated":          sum(1 for r in subset if r.hallucinated),
            "missed":                sum(1 for r in subset if r.missed),
            "hallucinated_and_missed":sum(1 for r in subset if r.hallucinated and r.missed),
            "instruction_bleed":     sum(1 for r in subset if r.instruction_bleed),
        }

        accurate = sum(1 for r in subset if r.taxonomy_label == "accurate")

        print(f"\n── {tech.upper()} ({total} encounters) ──")
        print(f"  Accurate:                  {accurate:>3}/{total}  ({100*accurate/total:.0f}%)")

        print(f"\n  Flag counts (non-exclusive):")
        for flag, count in flag_counts.items():
            print(f"    {flag:<28} {count:>3}/{total}")

        print(f"\n  Primary taxonomy label (Dao et al. Figure 4):")
        for label in ORDERED_LABELS:
            count = labels.get(label, 0)
            if count > 0:
                pct = 100 * count / total
                print(f"    {label:<28} {count:>3}  ({pct:.0f}%)")

        print(f"\n  Severity breakdown:")
        for sev in ["severe", "moderate", "minor", "none"]:
            count = severity.get(sev, 0)
            print(f"    {sev:<12} {count:>3}")

    if len(techniques) > 1:
        print(f"\n── CROSS-TECHNIQUE COMPARISON (Dao et al. Figure 4 categories) ──")
        header = f"  {'Label':<28}" + "".join(f"{t[:10]:>12}" for t in techniques)
        print(header)

        for label in ORDERED_LABELS:
            row = f"  {label:<28}"
            any_nonzero = False
            for tech in techniques:
                subset = [r for r in results if r.technique == tech]
                count = sum(1 for r in subset if r.taxonomy_label == label)
                total = len(subset)
                row += f"  {count}/{total}".rjust(10)
                if count > 0:
                    any_nonzero = True
            if label == "accurate" or any_nonzero:
                print(row)

    print(f"\n{'='*66}\n")


def export_csv(results: list[JudgeResult], path: str) -> None:
    """Export judge results to CSV for poster/analysis."""
    rows = []
    for r in results:
        rows.append({
            "encounter_id":           r.encounter_id,
            "technique":              r.technique,
            "taxonomy_label":         r.taxonomy_label,
            "accurate":               r.taxonomy_label == "accurate",
            "severity":               r.severity,
            "structural_failure_type":r.structural_failure_type or "none",
            "attempts":               r.attempts,
            "instruction_bleed":      r.instruction_bleed,
            "hallucinated":           r.hallucinated,
            "missed":                 r.missed,
            "overall_failure_type":   r.overall_failure_type,
            "judge_skipped":          r.judge_skipped,
            "reasoning":              r.reasoning,
            "hallucinated_examples":  "; ".join(r.hallucinated_examples),
            "missed_examples":        "; ".join(r.missed_examples),
            "instruction_bleed_examples": "; ".join(r.instruction_bleed_examples),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"CSV exported to: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge failure taxonomy for SOAP note generation pipelines"
    )
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="One or more pipeline result JSON files (e.g. gvr_results.json pe_results.json cd_results.json)"
    )
    parser.add_argument(
        "--transcripts", required=True,
        help="Path to clinicalnlp_taskB_test1.csv"
    )
    parser.add_argument(
        "--output", default="judge_results.json",
        help="Output JSON path (default: judge_results.json)"
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Limit to first N records per technique (for testing)"
    )
    parser.add_argument(
        "--technique", nargs="+", default=None,
        help="Override technique names (must match order of --results files)"
    )
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set. Check your .env file.")
        sys.exit(1)

    if not os.path.exists(args.transcripts):
        print(f"Error: transcripts CSV not found: {args.transcripts}")
        sys.exit(1)

    print(f"Loading transcripts from: {args.transcripts}")
    transcripts = load_transcripts(args.transcripts)
    print(f"Loaded {len(transcripts)} transcripts")

    # ── Resumability: load checkpoint if present ──────────────────────
    checkpoint_path = args.output.replace(".json", "_checkpoint.json")
    all_judge_results: list[JudgeResult] = []
    completed_keys: set[tuple[str, str]] = set()

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            existing = json.load(f)
        all_judge_results = [JudgeResult(**r) for r in existing]
        completed_keys = {(r.encounter_id, r.technique) for r in all_judge_results}
        print(f"Resuming from checkpoint: {len(completed_keys)} records already judged")

    # ── Run judge over each results file ──────────────────────────────
    for i, results_path in enumerate(args.results):
        if not os.path.exists(results_path):
            print(f"Warning: results file not found, skipping: {results_path}")
            continue

        technique_override = args.technique[i] if (args.technique and i < len(args.technique)) else None

        print(f"\nLoading: {results_path}")
        pipeline_results = load_pipeline_results(results_path)

        if args.n:
            pipeline_results = pipeline_results[:args.n]

        print(f"Judging {len(pipeline_results)} encounters...")

        for pr in pipeline_results:
            enc_id = pr["encounter_id"]
            tech   = technique_override or pr.get("technique", "unknown")
            key    = (enc_id, tech)

            if key in completed_keys:
                print(f"  Skipping {enc_id} [{tech}] (already judged)")
                continue

            jr = judge_single(
                pipeline_result=pr,
                transcripts=transcripts,
                technique_override=technique_override,
            )
            all_judge_results.append(jr)
            completed_keys.add(key)

            # Incremental checkpoint save after every record
            with open(checkpoint_path, "w") as f:
                json.dump([asdict(r) for r in all_judge_results], f, indent=2)

    # ── Final outputs ─────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in all_judge_results], f, indent=2)
    print(f"\nJudge results saved to: {args.output}")

    csv_path = args.output.replace(".json", ".csv")
    export_csv(all_judge_results, csv_path)

    print_summary(all_judge_results)

    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint removed: {checkpoint_path}")


if __name__ == "__main__":
    main()
