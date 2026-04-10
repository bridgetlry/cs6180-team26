# Generate-Validate-Retry (GVR) Pipeline
### CS6180 Team 26 — Elisabeth Sluchak

Structured SOAP note extraction from doctor-patient transcripts using a generate-validate-retry pipeline with Pydantic schema enforcement on the [ACI-Bench](https://github.com/wyim/aci-bench) dataset.

---

## Overview

The GVR pipeline sends each transcript to `meta-llama/llama-3.1-8b-instruct` via OpenRouter, validates the JSON response against a Pydantic schema, and retries with targeted feedback if validation fails. Up to 3 LLM calls per encounter (1 initial + 2 retries).

**Key files:**
```
generate_validate_retry.py   — main pipeline
gvr_pydantic_schema.py       — SOAPNote Pydantic schema
gvr_prompt_templates.py      — initial + retry prompt builders
aci_data_loader.py           — shared ACI-Bench data loader
evaluation/evaluate_fullnote.py  — ROUGE scoring script
```

---

## Setup

**1. Clone the repo and install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Create a `.env` file in the project root:**
```
OPENROUTER_API_KEY=your_key_here
```

**3. Download ACI-Bench data** from [figshare](https://doi.org/10.6084/m9.figshare.22494601) and place in `data/`:
```
data/
  clinicalnlp_taskB_test1.csv
  clinicalnlp_taskB_test1_metadata.csv
```

---

## Running the Pipeline

**Single encounter (first record, for testing):**
```bash
python generate_validate_retry.py single
```

**Small batch (first N encounters):**
```bash
python generate_validate_retry.py batch --n 5
```

**Full test set (all 40 encounters):**
```bash
python generate_validate_retry.py batch
```

The pipeline saves results incrementally — if interrupted, rerunning will resume from where it left off.

**Outputs:**
- `gvr_results.json` — full structured output per encounter including typed fields, failure info, and working_space reasoning
- `gvr_results_predictions.csv` — plain-text generated notes for evaluation (4 columns matching paper format: `Dialogues`, `Reference Summaries`, `note`, `encounter_id`)

---

## Running Evaluation

Evaluation computes ROUGE-1, ROUGE-2, ROUGE-L at full-note and division level using the ACI-Bench evaluation script.

**Full 40 encounters:**
```bash
python evaluation/evaluate_fullnote.py \
  data/clinicalnlp_taskB_test1.csv \
  gvr_results_predictions.csv \
  data/clinicalnlp_taskB_test1_metadata.csv
```

**Filtered subset (e.g. n5):**
```bash
python evaluation/evaluate_fullnote.py \
  data/clinicalnlp_taskB_test1_n5.csv \
  baselines/predictions/gvr_results_n5_predictions.csv \
  data/clinicalnlp_taskB_test1_metadata_n5.csv
```

Results saved to `results/<predictions_filename>.json`.

**Note:** BERTScore and BLEURT are currently stubbed (returns 0.0) due to a Python 3.13/DeBERTa tokenizer incompatibility. ROUGE scores are fully functional. MEDCON requires QuickUMLS setup (see below).

---

## Evaluating Against Paper Baselines

The paper's ChatGPT and GPT-4 baseline predictions are in:
```
baselines/predictions/
  test1ChatGPT_.csv
  test1GPT-4_.csv
  test1Text-Davinci-002_.csv
  test1Text-Davinci-003_.csv
```

To score a baseline against gold:
```bash
python evaluation/evaluate_fullnote.py \
  data/clinicalnlp_taskB_test1.csv \
  baselines/predictions/test1ChatGPT_.csv \
  data/clinicalnlp_taskB_test1_metadata.csv
```

---

## Schema

The `SOAPNote` schema has two layers:

**ACI-Bench text divisions (required):**
- `chief_complaint` — primary reason for visit
- `subjective` — HPI, PMH, medications, social/family history
- `objective_exam` — physical exam findings
- `objective_results` — labs, imaging (empty string if none)
- `assessment_and_plan` — diagnosis and treatment plan

**Typed clinical fields (optional, novel contribution):**
- Vital signs: `bp_systolic`, `bp_diastolic`, `heart_rate`, `o2_saturation`, `temperature_f`, `respiratory_rate`
- Lab values: `hemoglobin_a1c`, `blood_glucose`
- Demographics: `patient_age`, `patient_gender`
- Symptoms: `pain_severity`, `pain_location`, `symptom_duration`, `associated_symptoms`
- History: `past_medical_history`, `current_medications`, `allergies`, `medication_count`
- Physical measurements: `muscle_strength_right/left`, `wound_length/width_cm`, etc.

All optional fields default to `null` if not mentioned in the transcript.

---

## MEDCON Setup (Optional)

MEDCON requires QuickUMLS with the UMLS 2022AA metathesaurus. If configured:

1. Download UMLS 2022AA from [NLM](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html)
2. Place `MRCONSO.RRF` and `MRSTY.RRF` in `resources/`
3. Run: `python -m quickumls.install resources/ resources/des`
4. In `evaluation/evaluate_fullnote.py`, replace the UMLS stub with: `from UMLS_evaluation import *`

---

## Results (Full Test Set, 40 Encounters)

| Metric | GVR (Llama-3.1-8B) | ChatGPT baseline |
|---|---|---|
| ROUGE-1 | 0.4082 | 0.4744 |
| ROUGE-2 | 0.1801 | 0.1901 |
| ROUGE-L | 0.2675 | 0.4247 |

**Pipeline stats:** 39/40 successful (97.5%), 100% first-pass rate, 1 json_parse failure (D2N117, aci subset).

**By division (ROUGE-1):**
- objective_exam: 0.5288 (strongest)
- objective_results: 0.3911
- subjective: 0.3699
- assessment_and_plan: 0.3049 (weakest)
