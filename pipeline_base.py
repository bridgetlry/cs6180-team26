"""
pipeline_base.py
────────────────
Shared infrastructure for all three SOAP note generation techniques:
  - GVR                  (generate_validate_retry.py)
  - Prompt Engineering   (prompt_engineering.py)
  - Constrained Decoding (constrained_decoding.py)

To implement a new technique, subclass SOAPPipeline and implement run_pipeline().
Everything else — LLM calls, result structure, text formatting — is inherited.
"""

from __future__ import annotations

import os
import time
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Optional

from dotenv import load_dotenv

from aci_data_loader import ACIEncounter
from gvr_pydantic_schema import SOAPNote

load_dotenv()

# ─────────────────────────────────────────────
# SHARED CONFIGURATION
# ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL              = "meta-llama/llama-3.1-8b-instruct"
API_URL            = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE        = 0.0    # deterministic for reproducibility
DELAY_SECONDS      = 0.5    # between API calls — increase if rate limited


# ─────────────────────────────────────────────
# SHARED RESULT DATACLASS
# ─────────────────────────────────────────────
@dataclass
class PipelineResult:
    """
    Uniform output for all three techniques.
    The batch runner, evaluation scripts, and failure taxonomy
    all consume this — technique implementations must not change its shape.
    """
    # identity
    encounter_id:    str
    dataset_subset:  str            # virtassist | virtscribe | aci
    technique:       str            # "gvr" | "prompt_engineering" | "constrained_decoding"

    # outcome
    success:         bool
    attempts:        int            # total LLM calls made (1-3 for GVR; always 1 for others)
    soap_note:       Optional[dict] # validated SOAPNote as dict, or None if failed

    # failure tracking (Dao et al. failure taxonomy)
    final_error:     Optional[str]
    failure_type:    Optional[str]  # json_parse | pydantic | unexpected | None
    failed_fields:   list = dc_field(default_factory=list)

    # semantic flags
    plan_is_inferred: Optional[bool] = None

    # ground truth from metadata (populated from ACIEncounter, not from LLM output)
    chief_complaint_gt: Optional[str] = None
    patient_age_gt:     Optional[str] = None
    patient_gender_gt:  Optional[str] = None

    # reference note (gold standard, for ROUGE/BERTScore/MEDCON evaluation)
    reference_note:  Optional[str] = None


# ─────────────────────────────────────────────
# SHARED LLM WRAPPER
# ─────────────────────────────────────────────
def call_llm(
    prompt: str,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = 3000,
    delay: float = DELAY_SECONDS,
) -> str:
    """
    Send a single prompt to OpenRouter and return the response text.

    All three techniques call this. If you need to swap the model or
    adjust parameters for your technique, pass them as keyword arguments —
    don't modify the defaults here (those are shared).

    Raises:
        requests.HTTPError: on non-2xx responses (e.g. rate limit, auth failure)
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    time.sleep(delay)
    return response.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────
# SHARED TEXT FORMATTER
# ─────────────────────────────────────────────
def soap_note_to_text(soap: SOAPNote, encounter_id: str) -> str:
    """
    Convert a validated SOAPNote into plain-text clinical note format.

    Section headers match the gold reference notes and ACI-Bench
    sectiontagger.py patterns so evaluation scores correctly:
      CHIEF COMPLAINT
      HISTORY OF PRESENT ILLNESS   <- soap.subjective
      PHYSICAL EXAM                <- soap.objective_exam
      RESULTS                      <- soap.objective_results (omitted if empty)
      ASSESSMENT AND PLAN          <- soap.assessment_and_plan
    """
    parts = [
        f"CHIEF COMPLAINT\n\n{soap.chief_complaint}",
        f"HISTORY OF PRESENT ILLNESS\n\n{soap.subjective}",
        f"PHYSICAL EXAM\n\n{soap.objective_exam}",
    ]

    if soap.objective_results and soap.objective_results.strip():
        parts.append(f"RESULTS\n\n{soap.objective_results}")

    parts.append(f"ASSESSMENT AND PLAN\n\n{soap.assessment_and_plan}")

    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────
class SOAPPipeline(ABC):
    """
    Base class for all three SOAP note generation techniques.

    Subclasses must implement run_pipeline(). They get call_llm()
    and soap_note_to_text() for free, and must return a PipelineResult
    so the shared batch runner and evaluation scripts work unchanged.

    Minimal subclass skeleton:

        class MyTechniquePipeline(SOAPPipeline):
            TECHNIQUE_NAME = "my_technique"

            def run_pipeline(self, encounter, max_retries=0):
                try:
                    raw  = self.call_llm(your_prompt)
                    soap = SOAPNote(**your_parsed_output)
                    return self._success(encounter, soap, attempts=1)
                except Exception as e:
                    return self._failure(encounter, str(e), "unexpected", attempts=1)
    """

    # Set this in each subclass. It gets stamped into every PipelineResult
    # so the failure taxonomy and evaluation can group by technique.
    TECHNIQUE_NAME: str = "base"

    def call_llm(self, prompt: str, **kwargs) -> str:
        """Delegates to the module-level call_llm() so subclasses don't need a separate import."""
        return call_llm(prompt, **kwargs)

    def soap_note_to_text(self, soap: SOAPNote, encounter_id: str) -> str:
        """Delegates to the module-level soap_note_to_text()."""
        return soap_note_to_text(soap, encounter_id)

    @abstractmethod
    def run_pipeline(self, encounter: ACIEncounter, max_retries: int = 0) -> PipelineResult:
        """
        Run the technique on a single encounter and return a PipelineResult.

        Args:
            encounter:   One ACIEncounter from load_test1_encounters().
            max_retries: Additional attempts after the first failure.
                         GVR uses 2 (3 total calls). Prompt-eng and
                         constrained decoding typically use 0.

        Returns:
            A fully populated PipelineResult.
        """
        ...

    # ── Convenience constructors ──────────────────────────────────────────
    # Use _success() and _failure() inside run_pipeline() instead of
    # constructing PipelineResult by hand — they fill in all the shared
    # fields automatically from the encounter object.

    def _success(
        self,
        encounter: ACIEncounter,
        soap: SOAPNote,
        attempts: int,
        failed_fields: list | None = None,
    ) -> PipelineResult:
        """Build a successful PipelineResult from a validated SOAPNote."""
        return PipelineResult(
            encounter_id       = encounter.encounter_id,
            dataset_subset     = encounter.dataset,
            technique          = self.TECHNIQUE_NAME,
            success            = True,
            attempts           = attempts,
            soap_note          = soap.model_dump(),
            final_error        = None,
            failure_type       = None,
            failed_fields      = failed_fields or [],
            plan_is_inferred   = getattr(soap, "plan_is_inferred", None),
            chief_complaint_gt = encounter.chief_complaint,
            patient_age_gt     = encounter.patient_age,
            patient_gender_gt  = encounter.patient_gender,
            reference_note     = encounter.reference_note,
        )

    def _failure(
        self,
        encounter: ACIEncounter,
        error: str,
        failure_type: str,
        attempts: int,
        failed_fields: list | None = None,
    ) -> PipelineResult:
        """Build a failed PipelineResult."""
        return PipelineResult(
            encounter_id       = encounter.encounter_id,
            dataset_subset     = encounter.dataset,
            technique          = self.TECHNIQUE_NAME,
            success            = False,
            attempts           = attempts,
            soap_note          = None,
            final_error        = error,
            failure_type       = failure_type,
            failed_fields      = failed_fields or [],
            plan_is_inferred   = None,
            chief_complaint_gt = encounter.chief_complaint,
            patient_age_gt     = encounter.patient_age,
            patient_gender_gt  = encounter.patient_gender,
            reference_note     = encounter.reference_note,
        )
