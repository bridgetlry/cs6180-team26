"""
simple_baseline.py
──────────────────
Replicates the ACI-Bench paper's minimal prompt setup using
meta-llama/llama-3.1-8b-instruct via OpenRouter.

The prompt is the same minimal instruction the paper authors used with GPT models:
  "summarize the conversation to generate a clinical note with four sections:
   HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN.
   The conversation is: <transcript>"

Post-processing matches the ChatGPT rules from Table S1 of the paper's
supplemental info — normalizes markdown/hash/colon header variants into
plain uppercase headers so sectiontagger.py can correctly split divisions.

No JSON schema, no Pydantic validation, no retry — raw free-text output only.
Direct apples-to-apples comparison of Llama-3.1-8B vs the paper's GPT baselines.

Run:
    python simple_baseline.py single
    python simple_baseline.py batch
    python simple_baseline.py batch --n 5
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from aci_data_loader import ACIEncounter, load_test1_encounters
from pipeline_base import SOAPPipeline, PipelineResult

load_dotenv()

DATA_CSV     = "data/clinicalnlp_taskB_test1.csv"
METADATA_CSV = "data/clinicalnlp_taskB_test1_metadata.csv"
RESULTS_PATH = "results/constrained_JSON_output/simple_baseline_results.json"


# ─────────────────────────────────────────────
# SIMPLE BASELINE PIPELINE
# ─────────────────────────────────────────────
class SimpleBaselinePipeline(SOAPPipeline):
    """
    Minimal single-shot pipeline replicating the ACI-Bench paper prompt.

    Post-processing replicates the ChatGPT rules from Table S1 of the
    paper's supplemental info so sectiontagger.py can split divisions correctly.

    Post-processed text is stored in soap_note as {"raw": "<text>"} so the
    existing PipelineResult serialization works unchanged.
    """

    TECHNIQUE_NAME = "simple_baseline"

    # The four sections the paper's post-processing normalizes
    _SECTIONS = [
        "HISTORY OF PRESENT ILLNESS",
        "PHYSICAL EXAM",
        "RESULTS",
        "ASSESSMENT AND PLAN",
    ]

    def _build_prompt(self, encounter: ACIEncounter) -> str:
        return (
            "summarize the conversation to generate a clinical note with four sections: "
            "HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN. "
            f"The conversation is: {encounter.transcript}"
        )

    def _postprocess(self, text: str) -> str:
        """
        Apply the ChatGPT post-processing rules from Table S1 of the ACI-Bench
        supplemental info so sectiontagger.py correctly splits note divisions.

        Order matters — more specific patterns (bold, hash) before plain replacement.
        Handles: **SECTION**, ##SECTION, #SECTION:, #SECTION, <SECTION>, SECTION:, SECTION
        """
        text = text.replace("\n", " ").strip()
        for division in self._SECTIONS:
            text = text.replace(f"**{division}**", f"\n{division}\n")
            text = text.replace(f"##{division}",   f"\n{division}\n")
            text = text.replace(f"#{division}:",   f"\n{division}\n")
            text = text.replace(f"#{division}",    f"\n{division}\n")
            text = text.replace(f"<{division}>",   f"\n{division}\n")
            text = text.replace(f"{division}:",    f"\n{division}\n")
            text = text.replace(division,          f"\n{division}\n")
        return text.strip()

    def run_pipeline(
        self,
        encounter: ACIEncounter,
        max_retries: int = 0,
    ) -> PipelineResult:
        try:
            raw       = self.call_llm(self._build_prompt(encounter))
            processed = self._postprocess(raw)
            return PipelineResult(
                encounter_id       = encounter.encounter_id,
                dataset_subset     = encounter.dataset,
                technique          = self.TECHNIQUE_NAME,
                success            = True,
                attempts           = 1,
                soap_note          = {"raw": processed},
                final_error        = None,
                failure_type       = None,
                failed_fields      = [],
                plan_is_inferred   = None,
                chief_complaint_gt = encounter.chief_complaint,
                patient_age_gt     = encounter.patient_age,
                patient_gender_gt  = encounter.patient_gender,
                reference_note     = encounter.reference_note,
            )
        except Exception as e:
            return PipelineResult(
                encounter_id       = encounter.encounter_id,
                dataset_subset     = encounter.dataset,
                technique          = self.TECHNIQUE_NAME,
                success            = False,
                attempts           = 1,
                soap_note          = None,
                final_error        = str(e),
                failure_type       = "unexpected",
                failed_fields      = [],
                plan_is_inferred   = None,
                chief_complaint_gt = encounter.chief_complaint,
                patient_age_gt     = encounter.patient_age,
                patient_gender_gt  = encounter.patient_gender,
                reference_note     = encounter.reference_note,
            )


# ─────────────────────────────────────────────
# PREDICTIONS CSV — bypasses batch_runner's
# results_to_predictions() which expects a
# structured SOAPNote dict, not raw text
# ─────────────────────────────────────────────
def save_predictions(
    results: list[PipelineResult],
    output_path: str,
    encounters: list[ACIEncounter],
) -> None:
    """
    Write predictions CSV in the 4-column ACI-Bench format so
    evaluate_fullnote.py works on it unchanged:
        Dialogues, Reference Summaries, note, encounter_id
    """
    enc_index = {enc.encounter_id: enc for enc in encounters}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Dialogues", "Reference Summaries", "note", "encounter_id"]
        )
        writer.writeheader()
        for r in results:
            enc      = enc_index.get(r.encounter_id)
            raw_note = (r.soap_note or {}).get("raw", "") if r.success else ""
            writer.writerow({
                "Dialogues":           enc.transcript if enc else "",
                "Reference Summaries": enc.reference_note if enc else "",
                "note":                raw_note,
                "encounter_id":        r.encounter_id,
            })

    print(f"Saved {len(results)} predictions to {output_path}")


# ─────────────────────────────────────────────
# BATCH RUNNER — mirrors batch_runner.run_batch
# but uses our local save_predictions
# ─────────────────────────────────────────────
def run_batch(
    pipeline: SimpleBaselinePipeline,
    encounters: list[ACIEncounter],
    save_path: str = RESULTS_PATH,
) -> list[PipelineResult]:
    results: list[PipelineResult] = []
    completed_ids: set[str] = set()

    # Resume from existing file if present
    if os.path.exists(save_path):
        with open(save_path) as f:
            results = [PipelineResult(**r) for r in json.load(f)]
        completed_ids = {r.encounter_id for r in results}
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} encounters already done in {save_path}")

    for i, enc in enumerate(encounters):
        if enc.encounter_id in completed_ids:
            print(f"[{i+1}/{len(encounters)}] Skipping {enc.encounter_id} (already done)")
            continue

        print(f"\n[{i+1}/{len(encounters)}] Processing {enc.encounter_id} ...")
        result = pipeline.run_pipeline(enc)
        results.append(result)

        status = "OK" if result.success else "FAIL"
        print(f"  [{status}]")

        # Incremental save after every record
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump([dataclasses.asdict(r) for r in results], f, indent=2)

        pred_path = save_path.replace(".json", "_predictions.csv")
        save_predictions(results, pred_path, encounters)

    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    n = None
    if "--n" in sys.argv:
        try:
            n = int(sys.argv[sys.argv.index("--n") + 1])
        except (IndexError, ValueError):
            print("Usage: python simple_baseline.py [single|batch] [--n N]")
            sys.exit(1)

    encounters = load_test1_encounters(DATA_CSV, METADATA_CSV)
    pipeline   = SimpleBaselinePipeline()

    if mode == "single":
        enc = encounters[0]
        print(f"Running single record: {enc.encounter_id}")
        result = pipeline.run_pipeline(enc)
        status = "Success" if result.success else "Failed"
        print(f"\n[{status}]")
        if result.success:
            print("\n--- Generated note (post-processed) ---")
            print(result.soap_note["raw"])
        else:
            print(f"Error: {result.final_error}")

    elif mode == "batch":
        if n is not None:
            encounters = encounters[:n]
        save_path = (
            RESULTS_PATH.replace(".json", f"_n{len(encounters)}.json") if n
            else RESULTS_PATH
        )
        print(f"Running simple baseline on {len(encounters)} encounters ...")
        results = run_batch(pipeline, encounters, save_path=save_path)
        successes = sum(1 for r in results if r.success)
        print(f"\nDone: {successes}/{len(results)} successful")
        print(f"Results:     {save_path}")
        print(f"Predictions: {save_path.replace('.json', '_predictions.csv')}")

    else:
        print("Usage: python simple_baseline.py [single|batch] [--n N]")
        print("  single      — run on first encounter only (quick test)")
        print("  batch       — run on all 40 encounters")
        print("  batch --n 5 — run on first 5 encounters only")


if __name__ == "__main__":
    main()
