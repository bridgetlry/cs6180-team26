"""
constrained_decoding.py
───────────────────────
Constrained decoding technique using `instructor` to enforce Pydantic schema
at the token generation level.

Only technique-specific logic lives here. Everything shared
(LLM calls, PipelineResult, batch runner) is in pipeline_base.py / batch_runner.py.

Setup:
    pip install instructor openai
    Set OPENROUTER_API_KEY in your .env file.

Run:
    python constrained_decoding.py single
    python constrained_decoding.py batch
    python constrained_decoding.py batch --n 5
"""

from __future__ import annotations

import os

import instructor
from openai import OpenAI
from dotenv import load_dotenv

from aci_data_loader import ACIEncounter
from gvr_pydantic_schema import SOAPNote
from pipeline_base import SOAPPipeline, PipelineResult

load_dotenv()

# ─────────────────────────────────────────────
# CONSTRAINED DECODING PIPELINE
# ─────────────────────────────────────────────
class ConstrainedDecodingPipeline(SOAPPipeline):
    """
    Uses `instructor` to enforce the SOAPNote Pydantic schema at the token
    level — the model cannot produce output that fails schema validation.

    No retry logic needed: instructor either returns a valid SOAPNote or raises.
    """

    TECHNIQUE_NAME = "constrained_decoding"

    # Instructor wraps the OpenAI-compatible OpenRouter client
    _client = instructor.from_openai(
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    )

    def _build_prompt(self, encounter: ACIEncounter) -> str:
        """
        This is where your technique's prompt engineering lives.
        You have access to:
            encounter.transcript        — the full doctor-patient dialogue
            encounter.chief_complaint   — from ACI-Bench metadata (authoritative)
            encounter.patient_age       — from metadata
            encounter.patient_gender    — from metadata
        """
        return (
            f"Convert the following doctor-patient transcript into a structured SOAP note. "
            f"Use only information stated in the transcript.\n\n"
            f"TRANSCRIPT:\n{encounter.transcript}"
        )

    def run_pipeline(
        self,
        encounter: ACIEncounter,
        max_retries: int = 0,   # not used — instructor handles this internally
    ) -> PipelineResult:

        try:
            soap = self._client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct",
                response_model=SOAPNote,
                messages=[
                    {"role": "user", "content": self._build_prompt(encounter)}
                ],
            )
            return self._success(encounter, soap, attempts=1)

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
    from batch_runner import main
    main(
        ConstrainedDecodingPipeline(),
        default_results_path="results/cd_results.json",
        default_max_retries=0,  # constrained decoding does not retry
    )