from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class SOAPNote(BaseModel):
    """
    Structured SOAP note schema for ACI-Bench clinical note generation.

    Four required text divisions match ACI-Bench evaluation sections exactly:
      - subjective, objective_exam, objective_results, assessment_and_plan

    chief_complaint is required separately (always inferable from ACI-Bench transcripts).

    All typed numerical/clinical fields are Optional — set to None if not mentioned
    in the transcript. These are your novel contribution beyond the paper's schema.
    """

    # ── Required: ACI-Bench text divisions ────────────────────────────
    chief_complaint: str = Field(
        description="Primary reason for the visit, stated concisely"
    )
    subjective: str = Field(
        description=(
            "History of present illness, past medical history, medications, allergies, "
            "social history, family history — from verbal exam and patient report"
        )
    )
    objective_exam: str = Field(
        description=(
            "Physical examination findings from the visit — vitals, auscultation, "
            "palpation, inspection, neurological findings, etc."
        )
    )
    objective_results: str = Field(
        description=(
            "Lab results, imaging findings, diagnostic test results discussed during "
            "the visit. Use empty string if none mentioned."
        )
    )
    assessment_and_plan: str = Field(
        description=(
            "Clinician's diagnosis or clinical impression, plus next steps: "
            "treatments, medications, orders, referrals, and follow-up instructions"
        )
    )

    # ── Optional: typed boolean fields ────────────────────────────────
    is_urgent: Optional[bool] = Field(
        default=None,
        description="True if the visit involves an urgent or emergent concern"
    )
    smoker: Optional[bool] = Field(
        default=None,
        description="True if patient currently smokes"
    )
    alcohol_use: Optional[bool] = Field(
        default=None,
        description="True if patient reports alcohol use"
    )
    recreational_drugs: Optional[bool] = Field(
        default=None,
        description="True if patient reports recreational drug use"
    )

    # ── Optional: patient demographics ────────────────────────────────
    patient_age: Optional[int] = Field(
        default=None,
        description="Patient age in years"
    )
    patient_gender: Optional[str] = Field(
        default=None,
        description="Patient gender: male, female, other, or unknown"
    )

    # ── Optional: symptom descriptors ─────────────────────────────────
    symptom_duration: Optional[str] = Field(
        default=None,
        description="How long the chief complaint has been present"
    )
    pain_severity: Optional[float] = Field(
        default=None,
        description="Pain score on a 0–10 scale"
    )
    pain_quality: Optional[str] = Field(
        default=None,
        description="Description of pain character, e.g. sharp, dull, throbbing"
    )
    pain_location: Optional[str] = Field(
        default=None,
        description="Location of pain on the body"
    )
    associated_symptoms: Optional[str] = Field(
        default=None,
        description="Other symptoms mentioned alongside the chief complaint"
    )

    # ── Optional: vital signs ─────────────────────────────────────────
    bp_systolic: Optional[int] = Field(
        default=None,
        description="Systolic blood pressure in mmHg, e.g. 120 from 120/77"
    )
    bp_diastolic: Optional[int] = Field(
        default=None,
        description="Diastolic blood pressure in mmHg, e.g. 77 from 120/77"
    )
    heart_rate: Optional[int] = Field(
        default=None,
        description="Heart rate in BPM"
    )
    o2_saturation: Optional[float] = Field(
        default=None,
        description="Oxygen saturation as a percentage, e.g. 99.0"
    )
    respiratory_rate: Optional[int] = Field(
        default=None,
        description="Respiratory rate in breaths per minute"
    )
    temperature_f: Optional[float] = Field(
        default=None,
        description="Body temperature in Fahrenheit, e.g. 98.1"
    )

    # ── Optional: lab values ──────────────────────────────────────────
    hemoglobin_a1c: Optional[float] = Field(
        default=None,
        description="HbA1c percentage, e.g. 7.4"
    )
    blood_glucose: Optional[float] = Field(
        default=None,
        description="Blood glucose in mg/dL"
    )

    # ── Optional: physical exam measurements ─────────────────────────
    heart_rate_reported: Optional[str] = Field(
        default=None,
        description="Heart rate as reported verbally by patient or doctor"
    )
    systolic_murmur_grade: Optional[int] = Field(
        default=None,
        description="Systolic ejection murmur grade out of 6, e.g. 2 for a 2/6 murmur"
    )
    wound_length_cm: Optional[float] = Field(
        default=None,
        description="Wound length in cm (convert from inches if needed: 1 in = 2.54 cm)"
    )
    wound_width_cm: Optional[float] = Field(
        default=None,
        description="Wound width in cm"
    )
    kidney_stone_size_mm: Optional[float] = Field(
        default=None,
        description="Kidney stone size in mm"
    )
    bladder_volume_prevoid_ml: Optional[float] = Field(
        default=None,
        description="Bladder volume before voiding in mL"
    )
    bladder_volume_postvoid_ml: Optional[float] = Field(
        default=None,
        description="Bladder volume after voiding in mL"
    )
    muscle_strength_right: Optional[int] = Field(
        default=None,
        description="Right-side muscle strength grade out of 5"
    )
    muscle_strength_left: Optional[int] = Field(
        default=None,
        description="Left-side muscle strength grade out of 5"
    )
    bsa: Optional[float] = Field(
        default=None,
        description="Body surface area in m²"
    )

    # ── Optional: history fields ──────────────────────────────────────
    past_medical_history: Optional[str] = Field(default=None)
    current_medications: Optional[str] = Field(default=None)
    allergies: Optional[str] = Field(default=None)
    family_history: Optional[str] = Field(default=None)
    occupation: Optional[str] = Field(default=None)
    medication_count: Optional[int] = Field(
        default=None,
        description="Number of distinct medications mentioned in the visit"
    )

    # ── Validators ────────────────────────────────────────────────────

    @field_validator("chief_complaint", "subjective", "objective_exam", "assessment_and_plan")
    @classmethod
    def no_empty_strings(cls, v):
        if not v or not v.strip():
            raise ValueError("Field must not be empty or whitespace only")
        return v.strip()

    @field_validator("objective_results")
    @classmethod
    def normalize_objective_results(cls, v):
        # objective_results is legitimately empty in many encounters
        return v.strip() if v else ""

    @field_validator("patient_age")
    @classmethod
    def age_must_be_positive(cls, v):
        if v is not None and not (0 <= v <= 130):
            raise ValueError(f"patient_age must be between 0 and 130, got {v}")
        return v

    @field_validator("pain_severity")
    @classmethod
    def pain_severity_range(cls, v):
        if v is not None and not (0.0 <= v <= 10.0):
            raise ValueError(f"pain_severity must be between 0.0 and 10.0, got {v}")
        return v

    @field_validator("temperature_f")
    @classmethod
    def temperature_range(cls, v):
        if v is not None and not (90.0 <= v <= 108.0):
            raise ValueError(f"temperature_f must be between 90.0 and 108.0°F, got {v}")
        return v

    @field_validator("patient_gender")
    @classmethod
    def normalize_gender(cls, v):
        if v is not None:
            v = v.strip().lower()
            if v not in {"male", "female", "other", "unknown"}:
                raise ValueError(
                    f"patient_gender must be one of: male, female, other, unknown — got '{v}'"
                )
        return v

    @field_validator("medication_count")
    @classmethod
    def medication_count_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"medication_count must be non-negative, got {v}")
        return v

    @field_validator("bp_systolic")
    @classmethod
    def bp_systolic_range(cls, v):
        if v is not None and not (50 <= v <= 300):
            raise ValueError(f"bp_systolic must be between 50 and 300 mmHg, got {v}")
        return v

    @field_validator("bp_diastolic")
    @classmethod
    def bp_diastolic_range(cls, v):
        if v is not None and not (20 <= v <= 200):
            raise ValueError(f"bp_diastolic must be between 20 and 200 mmHg, got {v}")
        return v

    @field_validator("heart_rate")
    @classmethod
    def heart_rate_range(cls, v):
        if v is not None and not (20 <= v <= 300):
            raise ValueError(f"heart_rate must be between 20 and 300 BPM, got {v}")
        return v

    @field_validator("o2_saturation")
    @classmethod
    def o2_saturation_range(cls, v):
        if v is not None and not (50.0 <= v <= 100.0):
            raise ValueError(f"o2_saturation must be between 50.0 and 100.0%, got {v}")
        return v

    @field_validator("respiratory_rate")
    @classmethod
    def respiratory_rate_range(cls, v):
        if v is not None and not (4 <= v <= 60):
            raise ValueError(f"respiratory_rate must be between 4 and 60 breaths/min, got {v}")
        return v

    @field_validator("wound_length_cm", "wound_width_cm")
    @classmethod
    def wound_dimensions_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError(f"Wound dimensions must be positive, got {v}")
        return v

    @field_validator("kidney_stone_size_mm")
    @classmethod
    def stone_size_range(cls, v):
        if v is not None and not (0.0 < v <= 30.0):
            raise ValueError(f"kidney_stone_size_mm must be between 0 and 30 mm, got {v}")
        return v

    @field_validator("bladder_volume_prevoid_ml", "bladder_volume_postvoid_ml")
    @classmethod
    def bladder_volume_range(cls, v):
        if v is not None and not (0.0 <= v <= 1500.0):
            raise ValueError(f"Bladder volume must be between 0 and 1500 mL, got {v}")
        return v

    @field_validator("muscle_strength_right", "muscle_strength_left")
    @classmethod
    def muscle_strength_range(cls, v):
        if v is not None and v not in range(0, 6):
            raise ValueError(f"Muscle strength grade must be 0–5, got {v}")
        return v

    @field_validator("bsa")
    @classmethod
    def bsa_range(cls, v):
        if v is not None and not (0.5 <= v <= 3.5):
            raise ValueError(f"BSA must be between 0.5 and 3.5 m², got {v}")
        return v

    @field_validator("hemoglobin_a1c")
    @classmethod
    def a1c_range(cls, v):
        if v is not None and not (3.0 <= v <= 20.0):
            raise ValueError(f"hemoglobin_a1c must be between 3.0 and 20.0, got {v}")
        return v

    @field_validator("blood_glucose")
    @classmethod
    def glucose_range(cls, v):
        if v is not None and not (20.0 <= v <= 800.0):
            raise ValueError(f"blood_glucose must be between 20.0 and 800.0 mg/dL, got {v}")
        return v

    @field_validator("systolic_murmur_grade")
    @classmethod
    def murmur_grade_range(cls, v):
        if v is not None and v not in range(1, 7):
            raise ValueError(f"systolic_murmur_grade must be between 1 and 6, got {v}")
        return v