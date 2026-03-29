# cs6180-team26
**Organizing Unstructured Doctor-Patient Conversations Utilizing Different LLM Strategies**

Comparing three structured output generation techniques for clinical SOAP note extraction from ACI-Bench doctor-patient transcripts.

---

## Project structure

```
cs6180-team26/
├── pipeline_base.py             # SHARED — base class, PipelineResult, LLM wrapper
├── batch_runner.py              # SHARED — batch loop, saving, CLI, summary stats
├── aci_data_loader.py           # SHARED — loads ACI-Bench test1 encounters
│
├── generate_validate_retry.py   # Technique: GVR (Liz)
├── prompt_engineering.py        # Technique: Prompt Engineering (teammate)
├── constrained_decoding.py      # Technique: Constrained Decoding (teammate)
│
├── gvr_pydantic_schema.py       # Shared SOAPNote Pydantic schema
├── gvr_prompt_templates.py      # GVR prompt templates (initial + retry)
│
├── data/
│   ├── clinicalnlp_taskB_test1.csv
│   └── clinicalnlp_taskB_test1_metadata.csv
│
└── results/                     # Auto-created on first run
    ├── gvr_results.json
    ├── gvr_results_predictions.csv
    ├── pe_results.json
    ├── pe_results_predictions.csv
    ├── cd_results.json
    └── cd_results_predictions.csv
```

---

## Setup

**1. Clone the repo and install dependencies**
```bash
git clone https://github.com/bridgetlry/cs6180-team26.git
cd cs6180-team26
pip install -r requirements.txt
```

**2. Create a `.env` file with your OpenRouter API key**
```
OPENROUTER_API_KEY=your_key_here
```

**3. Add the ACI-Bench data files** to a `data/` directory:
```
data/clinicalnlp_taskB_test1.csv
data/clinicalnlp_taskB_test1_metadata.csv
```
Download from: https://doi.org/10.6084/m9.figshare.22494601

---

## Running your technique

All three technique scripts share the same CLI:

```bash
# Quick test — runs on the first encounter only
python <technique>.py single

# Full run — all 40 ACI-Bench test1 encounters
python <technique>.py batch

# Partial run — first N encounters (useful for debugging)
python <technique>.py batch --n 5
```

Replace `<technique>` with your script name:

| Technique | Script | Results saved to |
|---|---|---|
| GVR | `generate_validate_retry.py` | `results/gvr_results.json` |
| Prompt Engineering | `prompt_engineering.py` | `results/pe_results.json` |
| Constrained Decoding | `constrained_decoding.py` | `results/cd_results.json` |

Each run also writes a `_predictions.csv` alongside the JSON, in the format the ACI-Bench evaluation scripts expect.

Runs are **resumable** — if interrupted, rerunning will skip already-completed encounters.

---

## How to implement your technique

Your technique file only needs to do two things:

**1. Subclass `SOAPPipeline` and implement `run_pipeline()`**

```python
from pipeline_base import SOAPPipeline, PipelineResult
from aci_data_loader import ACIEncounter
from gvr_pydantic_schema import SOAPNote

class MyTechniquePipeline(SOAPPipeline):
    TECHNIQUE_NAME = "my_technique"   # "prompt_engineering" or "constrained_decoding"

    def run_pipeline(self, encounter: ACIEncounter, max_retries: int = 0) -> PipelineResult:
        try:
            raw  = self.call_llm(your_prompt)
            soap = SOAPNote(**your_parsed_output)
            return self._success(encounter, soap, attempts=1)
        except Exception as e:
            return self._failure(encounter, str(e), "unexpected", attempts=1)
```

You get for free:
- `self.call_llm(prompt)` — sends to OpenRouter, handles auth and rate limiting
- `self._success(encounter, soap, attempts)` — builds a complete `PipelineResult`
- `self._failure(encounter, error, failure_type, attempts)` — builds a failed `PipelineResult`
- `self.soap_note_to_text(soap, encounter_id)` — converts validated SOAPNote to ACI-Bench plain-text format

**2. Call `main()` at the bottom**

```python
if __name__ == "__main__":
    from batch_runner import main
    main(
        MyTechniquePipeline(),
        default_results_path="results/my_results.json",
        default_max_retries=0,
    )
```

That's it. The batch loop, incremental saving, predictions CSV, and summary stats are all handled automatically.

---

## The shared SOAPNote schema

All three techniques validate against the same Pydantic model in `gvr_pydantic_schema.py`.
Key fields your prompt needs to produce:

| Field | Type | Notes |
|---|---|---|
| `chief_complaint` | `str` | Required |
| `subjective` | `str` | Required — maps to HISTORY OF PRESENT ILLNESS |
| `objective_exam` | `str` | Required — maps to PHYSICAL EXAM |
| `objective_results` | `Optional[str]` | Maps to RESULTS section; omit if empty |
| `assessment_and_plan` | `str` | Required |
| `allergies` | `str` | Required (use "NKDA" or "None known" if not mentioned) |
| `patient_age` | `Optional[int]` | Numeric, not string |
| `is_urgent` | `Optional[bool]` | Boolean, not string |

> **Important:** The model must return these exact field names. If you see Pydantic
> validation errors, the most common cause is the model inventing its own schema
> (nested keys, different names) instead of following yours. Make your prompt
> explicitly list the required field names.

---

## Constrained decoding note

The `instructor` library must be initialized in **JSON mode** to avoid tool-call
compatibility issues with `llama-3.1-8b-instruct` via OpenRouter:

```python
import instructor
from instructor import Mode
from openai import OpenAI

client = instructor.from_openai(
    OpenAI(base_url="https://openrouter.ai/api/v1", api_key=...),
    mode=Mode.JSON   # required — TOOLS mode causes multiple tool call errors
)
```

---

## Shared data model

Each encounter loaded by `aci_data_loader.py` gives you:

```python
encounter.encounter_id      # e.g. "D2N088"
encounter.transcript        # full doctor-patient dialogue
encounter.reference_note    # gold human-written SOAP note (for evaluation)
encounter.chief_complaint   # from ACI-Bench metadata (authoritative ground truth)
encounter.patient_age       # from metadata
encounter.patient_gender    # from metadata
encounter.dataset           # "aci" | "virtassist" | "virtscribe"
```

---

## Output format

Every technique produces a `PipelineResult` with a consistent shape:

```python
result.encounter_id       # which encounter
result.technique          # "gvr" | "prompt_engineering" | "constrained_decoding"
result.success            # True / False
result.attempts           # how many LLM calls were made
result.soap_note          # validated SOAPNote as dict (None if failed)
result.failure_type       # "json_parse" | "pydantic" | "unexpected" | None
result.failed_fields      # list of Pydantic field names that failed
result.reference_note     # gold note for evaluation
```

This uniform shape means the evaluation scripts and failure taxonomy analysis
work identically across all three techniques.
