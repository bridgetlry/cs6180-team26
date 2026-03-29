# import json
# import time
# import requests
# import os
# from dataclasses import dataclass, field as dc_field
# from typing import Optional
# from pydantic import ValidationError
# from dotenv import load_dotenv
#
# from aci_data_loader import ACIEncounter, load_test1_encounters, save_predictions
# from gvr_pydantic_schema import SOAPNote
# from gvr_prompt_templates import build_initial_prompt, build_retry_prompt, format_validation_error
#
# load_dotenv()
#
# # ─────────────────────────────────────────────
# # CONFIGURATION
# # ─────────────────────────────────────────────
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# MODEL              = "meta-llama/llama-3.1-8b-instruct"
# API_URL            = "https://openrouter.ai/api/v1/chat/completions"
# MAX_RETRIES        = 2       # max 3 total LLM calls per record (Dao et al.)
# DELAY_SECONDS      = 0.5     # between calls — increase if rate limited
# TEMPERATURE        = 0.0     # deterministic for reproducibility
# RESULTS_PATH       = "results/gvr_results.json"
#
# DATA_CSV      = "data/clinicalnlp_taskB_test1.csv"
# METADATA_CSV  = "data/clinicalnlp_taskB_test1_metadata.csv"
#
#
# # ─────────────────────────────────────────────
# # RESULT DATACLASS
# # ─────────────────────────────────────────────
# @dataclass
# class PipelineResult:
#     # identity
#     encounter_id: str
#     dataset_subset: str           # virtassist | virtscribe | aci
#
#     # outcome
#     success: bool
#     attempts: int                 # 1, 2, or 3
#     soap_note: Optional[dict]     # validated SOAPNote as dict, None if failed
#
#     # failure tracking (for failure taxonomy)
#     final_error: Optional[str]
#     failure_type: Optional[str]   # json_parse | pydantic | unexpected | None
#     failed_fields: list = dc_field(default_factory=list)
#
#     # semantic flags
#     plan_is_inferred: Optional[bool] = None
#
#     # ground truth from metadata
#     chief_complaint_gt: Optional[str] = None
#     patient_age_gt: Optional[str] = None
#     patient_gender_gt: Optional[str] = None
#
#     # reference note (for evaluation)
#     reference_note: Optional[str] = None
#
#
# # ─────────────────────────────────────────────
# # LLM WRAPPER
# # ─────────────────────────────────────────────
# def call_llm(prompt: str, temperature: float = TEMPERATURE) -> str:
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": MODEL,
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": temperature,
#         "max_tokens": 3000
#     }
#     response = requests.post(API_URL, headers=headers, json=payload)
#     response.raise_for_status()
#     return response.json()["choices"][0]["message"]["content"].strip()
#
#
# def parse_json_response(raw: str) -> dict:
#     """
#     Strip markdown fences if present, attempt repair if needed, then parse JSON.
#     """
#     clean = raw.strip()
#
#     # strip markdown fences
#     if clean.startswith("```"):
#         lines = clean.split("\n")
#         clean = "\n".join(lines[1:-1]).strip()
#
#     # attempt direct parse first
#     try:
#         return json.loads(clean)
#     except json.JSONDecodeError:
#         pass
#
#     # repair: find first { and last } and extract that substring
#     start = clean.find("{")
#     end   = clean.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         try:
#             return json.loads(clean[start:end+1])
#         except json.JSONDecodeError:
#             pass
#
#     raise json.JSONDecodeError("Could not parse JSON from response", clean, 0)
#
#
# def soap_note_to_text(soap: SOAPNote, encounter_id: str) -> str:
#     """
#     Convert a validated SOAPNote into a plain-text clinical note
#     in the format ACI-Bench evaluation expects.
#
#     Section headers match the gold reference notes and sectiontagger.py patterns:
#       CHIEF COMPLAINT
#       HISTORY OF PRESENT ILLNESS  (subjective)
#       PHYSICAL EXAM               (objective_exam)
#       RESULTS                     (objective_results)
#       ASSESSMENT AND PLAN         (assessment_and_plan)
#     """
#     parts = []
#
#     parts.append(f"CHIEF COMPLAINT\n\n{soap.chief_complaint}")
#     parts.append(f"HISTORY OF PRESENT ILLNESS\n\n{soap.subjective}")
#     parts.append(f"PHYSICAL EXAM\n\n{soap.objective_exam}")
#
#     if soap.objective_results and soap.objective_results.strip():
#         parts.append(f"RESULTS\n\n{soap.objective_results}")
#
#     parts.append(f"ASSESSMENT AND PLAN\n\n{soap.assessment_and_plan}")
#
#     return "\n\n".join(parts)
#
#
# # ─────────────────────────────────────────────
# # CORE PIPELINE
# # ─────────────────────────────────────────────
# def run_pipeline(
#     encounter: ACIEncounter,
#     max_retries: int = MAX_RETRIES
# ) -> PipelineResult:
#     """
#     Generate-validate-retry pipeline for a single ACIEncounter.
#     """
#     transcript   = encounter.transcript
#     encounter_id = encounter.encounter_id
#     prompt       = build_initial_prompt(transcript, encounter_id)
#
#     last_error        = None
#     failure_type      = None
#     all_failed_fields = []
#
#     for attempt in range(max_retries + 1):
#         try:
#             raw = call_llm(prompt)
#             time.sleep(DELAY_SECONDS)
#
#             # ── Layer 1: JSON validation ──────────────────────────────
#             try:
#                 parsed = parse_json_response(raw)
#             except json.JSONDecodeError as e:
#                 failure_type = "json_parse"
#                 last_error   = f"Invalid JSON: {str(e)}"
#                 print(f"  [{encounter_id}] Attempt {attempt+1} — JSON parse error")
#                 prompt = build_retry_prompt(transcript, encounter_id, raw, last_error)
#                 continue
#
#             # ── Layer 2: Pydantic schema enforcement ──────────────────
#             try:
#                 # inject encounter_id placeholder only if schema still requires scenario_id
#                 if "scenario_id" in SOAPNote.model_fields and "scenario_id" not in parsed:
#                     parsed["scenario_id"] = 0
#
#                 soap = SOAPNote(**parsed)
#
#                 return PipelineResult(
#                     encounter_id      = encounter_id,
#                     dataset_subset    = encounter.dataset,
#                     success           = True,
#                     attempts          = attempt + 1,
#                     soap_note         = soap.model_dump(),
#                     final_error       = None,
#                     failure_type      = None,
#                     failed_fields     = all_failed_fields,
#                     plan_is_inferred  = getattr(soap, "plan_is_inferred", None),
#                     chief_complaint_gt = encounter.chief_complaint,
#                     patient_age_gt    = encounter.patient_age,
#                     patient_gender_gt = encounter.patient_gender,
#                     reference_note    = encounter.reference_note,
#                 )
#
#             except ValidationError as e:
#                 failure_type  = "pydantic"
#                 failed_fields = list({err["loc"][0] for err in e.errors() if err["loc"]})
#                 all_failed_fields.extend(failed_fields)
#                 last_error    = format_validation_error(e)
#                 print(f"  [{encounter_id}] Attempt {attempt+1} — Pydantic error: {failed_fields}")
#                 prompt = build_retry_prompt(transcript, encounter_id, raw, last_error)
#                 continue
#
#         except Exception as e:
#             failure_type = "unexpected"
#             last_error   = f"Unexpected error: {str(e)}"
#             print(f"  [{encounter_id}] Attempt {attempt+1} — Unexpected: {e}")
#             break
#
#     # all retries exhausted
#     return PipelineResult(
#         encounter_id      = encounter_id,
#         dataset_subset    = encounter.dataset,
#         success           = False,
#         attempts          = max_retries + 1,
#         soap_note         = None,
#         final_error       = last_error,
#         failure_type      = failure_type,
#         failed_fields     = all_failed_fields,
#         plan_is_inferred  = None,
#         chief_complaint_gt = encounter.chief_complaint,
#         patient_age_gt    = encounter.patient_age,
#         patient_gender_gt = encounter.patient_gender,
#         reference_note    = encounter.reference_note,
#     )
#
#
# # ─────────────────────────────────────────────
# # BATCH RUNNER
# # ─────────────────────────────────────────────
# def run_batch(
#     encounters: list,
#     max_retries: int = MAX_RETRIES,
#     save_path: str = RESULTS_PATH
# ) -> list:
#     # Load existing results if resuming
#     results = []
#     completed_ids = set()
#     if os.path.exists(save_path):
#         with open(save_path) as f:
#             existing = json.load(f)
#         # Reconstruct as plain dicts (not dataclasses) for now — we only need them for saving/predictions
#         results = [PipelineResult(**r) for r in existing]
#         completed_ids = {r.encounter_id for r in results}
#         if completed_ids:
#             print(f"Resuming: found {len(completed_ids)} completed encounters in {save_path}")
#
#     for i, enc in enumerate(encounters):
#         if enc.encounter_id in completed_ids:
#             print(f"[{i+1}/{len(encounters)}] Skipping {enc.encounter_id} (already done)")
#             continue
#
#         print(f"\n[{i+1}/{len(encounters)}] Processing {enc.encounter_id}...")
#         result = run_pipeline(enc, max_retries=max_retries)
#         results.append(result)
#
#         status = "✅" if result.success else "❌"
#         print(f"  {status} attempts={result.attempts} | "
#               f"failed_fields={result.failed_fields} | "
#               f"plan_inferred={result.plan_is_inferred}")
#
#         _save_results(results, save_path)
#
#         predictions = _results_to_predictions(results)
#         save_predictions(predictions, save_path.replace(".json", "_predictions.csv"), encounters=encounters)
#
#     return results
#
#
# def _results_to_predictions(results: list) -> dict:
#     """
#     Convert pipeline results to {encounter_id: note_text} dict for evaluation.
#     Failed records get an empty string so evaluation still runs.
#     """
#     predictions = {}
#     for r in results:
#         if r.success and r.soap_note:
#             # reconstruct a minimal SOAPNote to convert to text
#             try:
#                 soap = SOAPNote(**r.soap_note)
#                 predictions[r.encounter_id] = soap_note_to_text(soap, r.encounter_id)
#             except Exception:
#                 predictions[r.encounter_id] = ""
#         else:
#             predictions[r.encounter_id] = ""
#     return predictions
#
#
# def _save_results(results: list, path: str):
#     """Serialize full pipeline results to JSON."""
#     import dataclasses
#     with open(path, "w") as f:
#         json.dump(
#             [dataclasses.asdict(r) for r in results],
#             f, indent=2
#         )
#
#
# # ─────────────────────────────────────────────
# # SUMMARY STATS
# # ─────────────────────────────────────────────
# def print_summary(results: list):
#     from collections import Counter
#     total      = len(results)
#     successes  = [r for r in results if r.success]
#     failures   = [r for r in results if not r.success]
#     first_pass = [r for r in successes if r.attempts == 1]
#     retried    = [r for r in successes if r.attempts > 1]
#     inferred   = [r for r in successes if r.plan_is_inferred]
#
#     print(f"\n{'='*50}")
#     print(f"PIPELINE SUMMARY — {total} encounters")
#     print(f"{'='*50}")
#     print(f"Overall pass rate:     {len(successes)}/{total} ({100*len(successes)/total:.1f}%)")
#     print(f"First-pass rate:       {len(first_pass)}/{total} ({100*len(first_pass)/total:.1f}%)")
#     print(f"Succeeded after retry: {len(retried)}/{total} ({100*len(retried)/total:.1f}%)")
#     print(f"Failed after retries:  {len(failures)}/{total} ({100*len(failures)/total:.1f}%)")
#     print(f"Plan inferred:         {len(inferred)}/{len(successes)} successes")
#
#     dist = Counter(r.attempts for r in results)
#     print(f"\nAttempt distribution:")
#     for k in sorted(dist):
#         print(f"  {k} attempt(s): {dist[k]} records")
#
#     if failures:
#         ftypes = Counter(r.failure_type for r in failures)
#         print(f"\nFailure types (unresolved after retries):")
#         for k, v in ftypes.most_common():
#             print(f"  {k}: {v}")
#
#     all_fields = []
#     for r in results:
#         all_fields.extend(r.failed_fields)
#     if all_fields:
#         print(f"\nMost common failed fields (across all attempts):")
#         for f, count in Counter(all_fields).most_common(5):
#             print(f"  {f}: {count}")
#
#     print(f"{'='*50}\n")
#
#
# # ─────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────
# if __name__ == "__main__":
#     import sys
#
#     mode = sys.argv[1] if len(sys.argv) > 1 else "single"
#
#     # optional: --n <number> limits batch size
#     n = None
#     if "--n" in sys.argv:
#         try:
#             n = int(sys.argv[sys.argv.index("--n") + 1])
#         except (IndexError, ValueError):
#             print("Usage: python generate_validate_retry.py batch --n 5")
#             sys.exit(1)
#
#     encounters = load_test1_encounters(DATA_CSV, METADATA_CSV)
#
#     if mode == "single":
#         enc = encounters[0]
#         print(f"Running single record: {enc.encounter_id}")
#         result = run_pipeline(enc)
#         status = "✅ Success" if result.success else "❌ Failed"
#         print(f"\n{status} | attempts={result.attempts}")
#         if result.success:
#             print(json.dumps(result.soap_note, indent=2))
#             print("\n--- Generated note text ---")
#             soap = SOAPNote(**result.soap_note)
#             print(soap_note_to_text(soap, result.encounter_id))
#         else:
#             print(f"Error: {result.final_error}")
#
#     elif mode == "batch":
#         if n is not None:
#             encounters = encounters[:n]
#         save_path = RESULTS_PATH.replace(".json", f"_n{len(encounters)}.json") if n else RESULTS_PATH
#         print(f"Running batch on {len(encounters)} encounters...")
#         results = run_batch(encounters, save_path=save_path)
#         print_summary(results)
#         print(f"Results saved to: {save_path}")
#         print(f"Predictions CSV:  {save_path.replace('.json', '_predictions.csv')}")
#
#     else:
#         print("Usage: python generate_validate_retry.py [single|batch] [--n N]")
#         print("  single      — run on first encounter only")
#         print("  batch       — run on all 40 encounters")
#         print("  batch --n 5 — run on first 5 encounters only")