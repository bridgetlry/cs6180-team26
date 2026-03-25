from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SOAPNote(BaseModel):
    """
    Shared schema for structured SOAP note extraction from doctor-patient transcripts.
    Used by all three pipeline techniques: prompt engineering, constrained decoding,
    and generate-validate-retry.

    Based on ACI-bench dataset structure. Evaluated against paired clinician notes.
    Architecture inspired by Dao et al. (2025) clinical extraction pipeline.
    """

    # ─────────────────────────────────────────────
    # WORKING SPACE — model reasoning scratchpad
    # ─────────────────────────────────────────────
    working_space: str = Field(
        description=(
            "Use this field FIRST to reason through the transcript before extracting values. "
            "Note: what is the chief complaint, any pain scores mentioned, patient age if stated, "
            "urgency indicators, whether a plan was explicitly stated or must be inferred, "
            "any ambiguous or scattered information. Be concise."
        )
    )

    # ─────────────────────────────────────────────
    # NARRATIVE SOAP SECTIONS (required)
    # ─────────────────────────────────────────────
    subjective: str = Field(
        description=(
            "Patient's own account: symptoms, chief complaint, history of present illness, "
            "past medical history, medications, allergies, family and social history."
        )
    )
    objective: str = Field(
        description=(
            "Clinically observed or explicitly reported findings only: vitals mentioned, "
            "physical exam findings, reported measurements. Do not infer."
        )
    )
    assessment: str = Field(
        description=(
            "Clinician's clinical impression or differential diagnosis, grounded in "
            "transcript only. Do not fabricate diagnoses not discussed."
        )
    )
    plan: str = Field(
        description=(
            "Next steps: tests ordered, treatments, medications, referrals, follow-up. "
            "If not explicitly stated, infer a reasonable plan from clinical context "
            "and set plan_is_inferred to true."
        )
    )
    plan_is_inferred: bool = Field(
        description=(
            "True if plan was inferred by the model from clinical context. "
            "False if plan was explicitly stated in the transcript."
        )
    )

    # ─────────────────────────────────────────────
    # REQUIRED EXTRACTED FIELDS
    # ─────────────────────────────────────────────
    chief_complaint: str = Field(
        description="Primary reason the patient is seeking care, in one sentence."
    )
    is_urgent: bool = Field(
        description=(
            "True if the case involves severe distress, emergency symptoms, "
            "or requires immediate attention based on transcript content."
        )
    )

    # ─────────────────────────────────────────────
    # TYPED OPTIONAL FIELDS
    # ─────────────────────────────────────────────
    patient_age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Patient age in years. Null if not mentioned."
    )
    pain_severity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Pain severity on a 0-10 scale if explicitly mentioned. Null if not mentioned."
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=35.0,
        le=42.0,
        description="Body temperature in Celsius if mentioned. Null if not mentioned."
    )
    smoker: Optional[bool] = Field(
        default=None,
        description="True if patient is a current smoker. Null if not mentioned."
    )
    alcohol_use: Optional[bool] = Field(
        default=None,
        description="True if alcohol use is mentioned. Null if not mentioned."
    )
    recreational_drugs: Optional[bool] = Field(
        default=None,
        description="True if recreational drug use is mentioned. Null if not mentioned."
    )

    # ─────────────────────────────────────────────
    # OPTIONAL STRING FIELDS
    # ─────────────────────────────────────────────
    patient_gender: Optional[str] = Field(
        default=None,
        description="Patient gender if mentioned. Null if not mentioned."
    )
    symptom_duration: Optional[str] = Field(
        default=None,
        description="How long symptoms have been present (e.g. '3 days', '2 weeks'). Null if not mentioned."
    )
    pain_quality: Optional[str] = Field(
        default=None,
        description="Character of pain (e.g. 'sharp', 'burning', 'dull'). Null if not mentioned."
    )
    pain_location: Optional[str] = Field(
        default=None,
        description="Location of pain if mentioned. Null if not mentioned."
    )
    associated_symptoms: Optional[str] = Field(
        default=None,
        description="Other symptoms mentioned alongside the chief complaint. Null if not mentioned."
    )
    past_medical_history: Optional[str] = Field(
        default=None,
        description="Relevant past medical conditions. Null if not mentioned."
    )
    current_medications: Optional[str] = Field(
        default=None,
        description="Medications the patient is currently taking. Null if not mentioned."
    )
    allergies: Optional[str] = Field(
        default=None,
        description="Known allergies. Null if not mentioned."
    )
    family_history: Optional[str] = Field(
        default=None,
        description="Relevant family medical history. Null if not mentioned."
    )
    occupation: Optional[str] = Field(
        default=None,
        description="Patient's occupation if mentioned. Null if not mentioned."
    )
    probable_diagnosis: Optional[str] = Field(
        default=None,
        description="Most likely diagnosis if clearly stated or strongly implied. Null if not mentioned."
    )

    # ─────────────────────────────────────────────
    # VALIDATORS
    # ─────────────────────────────────────────────
    @field_validator(
        "temperature", "pain_severity", "patient_age",
        "smoker", "alcohol_use", "recreational_drugs",
        mode="before"
    )
    @classmethod
    def coerce_null_strings(cls, v):
        """Convert common null-like strings to None before type coercion."""
        if isinstance(v, str) and v.strip().lower() in {
            "null", "none", "n/a", "na", "not mentioned",
            "not stated", "not applicable", "unknown", ""
        }:
            return None
        return v

    @field_validator("pain_severity", mode="before")
    @classmethod
    def coerce_pain_severity(cls, v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"pain_severity must be a number between 0 and 10, got: {v!r}")

    @field_validator("patient_age", mode="before")
    @classmethod
    def coerce_patient_age(cls, v):
        if v is None:
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            raise ValueError(f"patient_age must be an integer, got: {v!r}")

    @field_validator("temperature", mode="before")
    @classmethod
    def coerce_temperature(cls, v):
        if v is None:
            return None
        try:
            val = float(v)
            # convert Fahrenheit to Celsius if value is in Fahrenheit range
            if val > 42.0:
                val = round((val - 32) / 1.8, 1)
            return val
        except (TypeError, ValueError):
            raise ValueError(f"temperature must be a number, got: {v!r}")