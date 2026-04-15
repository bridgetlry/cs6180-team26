"""
pipeline_base.py
────────────────
Shared infrastructure for all three SOAP note generation techniques:
  - GVR                  (generate_validate_retry.py)
  - Prompt Engineering   (prompt_engineering.py)
  - Constrained Decoding (constrained_decoding.py)

This file is the foundation that all three techniques build on top of.
It provides four things so technique files don't have to reimplement them:
  1. PipelineResult  — a standardized "report card" dataclass every technique returns
  2. call_llm()      — the shared function that talks to the OpenRouter API
  3. soap_note_to_text() — converts a validated SOAPNote into plain-text clinical note format
  4. SOAPPipeline    — the abstract base class each technique must subclass

To implement a new technique, subclass SOAPPipeline and implement run_pipeline().
Everything else — LLM calls, result structure, text formatting — is inherited for free.
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

# standard library imports
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Optional

# third-party imports
import requests
from dotenv import load_dotenv

# local imports
from experiments.shared_pipeline_elements.aci_data_loader import ACIEncounter
from experiments.shared_pipeline_elements.pydantic_schema import SOAPNote

load_dotenv()

# ─────────────────────────────────────────────
# SHARED CONFIGURATION
# ─────────────────────────────────────────────
# All three techniques use the same model and API settings.
# If you need to adjust parameters for your technique, pass them
# as keyword arguments to call_llm() — don't modify the defaults here.

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL              = "meta-llama/llama-3.1-8b-instruct"
API_URL            = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE        = 0.0    # deterministic — same input always produces same output
DELAY_SECONDS      = 0.5    # pause between API calls — increase if hitting rate limits


# ─────────────────────────────────────────────
# PIPELINE RESULT
# ─────────────────────────────────────────────
# A standardized "report card" that every technique returns for each encounter.
# Having one shared shape means the batch runner, evaluation scripts, and
# failure taxonomy all work identically across GVR, prompt engineering,
# and constrained decoding — none of them need to know which technique ran.
#
# Use _success() and _failure() (defined in SOAPPipeline below) to construct
# these — don't build them by hand, those helpers fill in all shared fields
# automatically from the encounter object.

@dataclass
class PipelineResult:
    """
    Uniform output for all three techniques.
    The batch runner, evaluation scripts, and failure taxonomy
    all consume this — technique implementations must not change its shape.
    """
    # which encounter this result is for
    encounter_id:    str
    dataset_subset:  str            # virtassist | virtscribe | aci
    technique:       str            # "gvr" | "prompt_engineering" | "constrained_decoding"

    # did it work, and how many LLM calls did it take?
    success:         bool
    attempts:        int            # total LLM calls made (1-3 for GVR; always 1 for others)
    soap_note:       Optional[dict] # validated SOAPNote as dict, or None if failed

    # failure tracking — used by the Dao et al. failure taxonomy analysis
    final_error:     Optional[str]
    failure_type:    Optional[str]  # json_parse | pydantic | unexpected | None
    failed_fields:   list = dc_field(default_factory=list)

    # semantic flags
    plan_is_inferred: Optional[bool] = None

    # ground truth from ACI-Bench metadata — populated from the encounter object,
    # NOT from LLM output, so we always have the authoritative values for comparison
    chief_complaint_gt: Optional[str] = None
    patient_age_gt:     Optional[str] = None
    patient_gender_gt:  Optional[str] = None

    # the gold standard clinical note, used for ROUGE/BERTScore/MEDCON scoring
    reference_note:  Optional[str] = None


# ─────────────────────────────────────────────
# SHARED LLM WRAPPER
# ─────────────────────────────────────────────
# One function that handles all communication with the OpenRouter API.
# All three techniques call this through self.call_llm() on the SOAPPipeline class.
# Centralizing this means if we ever need to swap models or change API behavior,
# there's only one place to update.

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
    time.sleep(delay)  # rate limit buffer between calls
    return response.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────
# SHARED TEXT FORMATTER
# ─────────────────────────────────────────────
# After a SOAPNote is validated by Pydantic, it needs to be converted to plain
# text before evaluation. This function applies the exact section headers that
# sectiontagger.py (in evaluation/) uses to split notes into divisions.
# If the headers don't match exactly, the evaluation script can't split the
# note correctly and division-level scores will be wrong.

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

    # objective_results is legitimately empty in many encounters — omit the
    # section entirely rather than writing an empty RESULTS block
    if soap.objective_results and soap.objective_results.strip():
        parts.append(f"RESULTS\n\n{soap.objective_results}")

    parts.append(f"ASSESSMENT AND PLAN\n\n{soap.assessment_and_plan}")

    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────
# SOAPPipeline is the template (contract) that all three techniques must follow.
# It says: "if you want to be a technique in this project, you must implement
# run_pipeline() and return a PipelineResult."
#
# By inheriting from SOAPPipeline, each technique file gets call_llm(),
# soap_note_to_text(), _success(), and _failure() for free — it only needs
# to implement the technique-specific logic inside run_pipeline().
#
# Example — minimal subclass:
#
#     class MyTechniquePipeline(SOAPPipeline):
#         TECHNIQUE_NAME = "my_technique"
#
#         def run_pipeline(self, encounter, max_retries=0):
#             try:
#                 raw  = self.call_llm(your_prompt)
#                 soap = SOAPNote(**your_parsed_output)
#                 return self._success(encounter, soap, attempts=1)
#             except Exception as e:
#                 return self._failure(encounter, str(e), "unexpected", attempts=1)

class SOAPPipeline(ABC):
    """
    Base class for all three SOAP note generation techniques.

    Subclasses must implement run_pipeline(). They get call_llm()
    and soap_note_to_text() for free, and must return a PipelineResult
    so the shared batch runner and evaluation scripts work unchanged.
    """

    # Each subclass sets this to its technique name.
    # It gets stamped into every PipelineResult so the failure taxonomy
    # and evaluation scripts can group and compare results by technique.
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
        This is the only method each technique file must implement.

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
    # Call these inside run_pipeline() instead of constructing PipelineResult
    # by hand. They automatically populate all shared fields from the encounter
    # object so you only need to pass in what your technique produced.

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