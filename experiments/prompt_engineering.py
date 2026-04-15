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

from experiments.shared_pipeline_elements.aci_data_loader import ACIEncounter
from experiments.shared_pipeline_elements.pydantic_schema import SOAPNote
from experiments.shared_pipeline_elements.pipeline_base import SOAPPipeline, PipelineResult


# ─────────────────────────────────────────────
# PROMPT ENGINEERING PIPELINE
# ─────────────────────────────────────────────
class PromptEngineeringPipeline(SOAPPipeline):
    """
    Prompt-only technique: a carefully engineered prompt drives the model
    to produce valid structured output without retry or constrained decoding.

    This is the baseline against which GVR and constrained decoding are compared.
    """

    TECHNIQUE_NAME = "prompt_engineering"

    def _build_prompt(self, encounter: ACIEncounter) -> str:
        """
        TODO: Replace with your actual engineered prompt.

        This is where your technique's work lives — few-shot examples,
        chain-of-thought, output format instructions, etc.

        You have access to:
            encounter.transcript        — the full doctor-patient dialogue
            encounter.chief_complaint   — from ACI-Bench metadata (authoritative)
            encounter.patient_age       — from metadata
            encounter.patient_gender    — from metadata
        """
        return (
            f"Convert the following doctor-patient transcript into a structured SOAP note "
            f"as a JSON object. Use only information stated in the transcript.\n\n"
            f"TRANSCRIPT:\n{encounter.transcript}\n\n"
            f"Respond with valid JSON only. No preamble, no explanation."
        )

    def _parse_response(self, raw: str) -> dict:
        """
        TODO: Replace or extend with your own parsing strategy.

        Prompt engineering typically relies on the model following format
        instructions, so parsing may be simpler than GVR — but you can
        add light cleanup here if needed (e.g. stripping markdown fences).
        """
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1]).strip()
        return json.loads(clean)

    def run_pipeline(
        self,
        encounter: ACIEncounter,
        max_retries: int = 0,   # prompt-eng is single-shot by design
    ) -> PipelineResult:

        try:
            raw    = self.call_llm(self._build_prompt(encounter))
            parsed = self._parse_response(raw)

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
    from experiments.shared_pipeline_elements.batch_runner import main
    main(
        PromptEngineeringPipeline(),
        default_results_path="../results/constrained_JSON_output/pe_results.json",
        default_max_retries=0,  # single-shot by design
    )
