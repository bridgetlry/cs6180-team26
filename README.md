# **Small Model, Strict Schema**: Benchmarking Structured Clinical Note Extraction from Doctor-Patient Conversations via ACI-Bench Dataset 
_Generative AI (CS 6180)_

Comparing three structured output generation techniques for clinical SOAP note extraction from ACI-Bench doctor-patient transcripts.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [ACI-BENCH Data Source](#aci-bench-data-source)
    - [Files](#files)
    - [Dataset subsets](#dataset-subsets)
    - [Download Instructions](#download-instructions)
- [ACI-Bench Baseline](#aci-bench-baseline)
    - [How to score a baseline](#how-to-score-a-baseline)
    - [Reference scores](#reference-scores-yim-et-al-2023-table-6--test-set-1)
- [Setup](#setup)
- [Running an Individual Technique](#running-an-individual-technique)
- [How to Implement a New Technique](#how-to-implement-a-new-technique)
- [The Shared SOAPNote Schema](#the-shared-soapnote-schema)
- [Evaluation](#evaluation)
    - [Files](#files-1)
    - [Dependencies](#dependencies)
    - [How to run](#how-to-run)
    - [Metrics](#metrics)
    - [MEDCON - UMLS / QuickUMLS setup](#medcon---umls--quickumls-setup)
- [Note on Constrained Decoding](#note-on-constrained-decoding)
- [Shared Data Model](#shared-data-model)
- [Output Format](#output-format)
---
## Introduction
The clinical note extraction system converts doctor-patient conversations to organized SOAP notes*. The data comes from the ACI-Bench dataset, which contains 40 encounters as well as as "gold-standard" SOAP notes  that we used to evalute the success of our methods. Our team leveraged three new different approaches (generate-validate-retry, constrained decoding, and prompt engineering) that utilized the `llama-3.1-8b-instruct` model to process the dialogue and generate the proper documentation.

*SOAP notes consist of Subjective Information, Objective Information, Assessment, and Plan, and is often used by physicians to document interactions with patients.

### Generate-Validate-Retry (GVR)
The conversation is processed up to three times with feedback-injected retry prompts. This gives the LLM the opportunity to improve its output by parsing it as a JSON and validating with Pydantic and retrying the generation with this feedback.

### Constrained Decoding
The `instructor` library is used to create the client that takes in the `SOAPNote` schema which gives the model a structure to enforce when generating SOAP note for the given conversation. 

### Prompt Engineering
Our baseline approach consists of a carefully engineered prompt that is provided to the model to generate a valid output without retry or constrained decoding.

---

## Project Structure 
--> TODO: needs to be updated with new organization

```
cs6180-team26/
├── app.py                       # Streamlit front end application
│
├── pipeline_base.py             # SHARED — base class, PipelineResult, LLM wrapper
├── batch_runner.py              # SHARED — batch loop, saving, CLI, summary stats
├── aci_data_loader.py           # SHARED — loads ACI-Bench test1 encounters
│
├── simple_baseline.py           # Technique: replication of ACI-Bench research's pipeline
│
├── generate_validate_retry.py   # Technique: GVR (Liz)
├── prompt_engineering.py        # Technique: Prompt Engineering (Priyam)
├── constrained_decoding.py      # Technique: Constrained Decoding (Bridget)
│
├── gvr_pydantic_schema.py       # Shared SOAPNote Pydantic schema
├── gvr_prompt_templates.py      # GVR prompt templates (initial + retry)
│
├── baseline/...                 # See README_baselines.md for folder structure
│
├── data/
│   ├── clinicalnlp_taskB_test1.csv
│   └── clinicalnlp_taskB_test1_metadata.csv
│
├── diagrams/                    # Mermaid diagrams for overall architecture and individual approaches
│   ├── cd.mmd
│   ├── full_arch.mmd
│   ├── gvr.mmd
│   └── pe.mmd
│
├── evaluation/...               # See README_evaluation.md for folder structure
│
├── resources/
│   └── semantic_types.txt       # MEDCON documentation subject categories
│
└── results/                     
    ├── gvr_results.json
    ├── gvr_results_predictions.csv
    ├── pe_results.json
    ├── pe_results_predictions.csv
    ├── cd_results.json
    ├── cd_results_predictions.csv
    ├── llm_judge.py             # Script to run LLM as judge for typed field extraction, see file for how to run
    ├── judge_results_gvr.json
    ├── judge_results_gvr.csv
    ├── judge_results_pe.json
    ├── judge_results_pe.csv
    ├── judge_results_cd.json
    ├── judge_results_cd.csv

```

---

## ACI-Bench Data Source
ACI-Bench test set 1 data files used in ACI Bench Paper and for our additional validated output experiments. 

Downloaded from the ACI-Bench figshare repository:
[https://doi.org/10.6084/m9.figshare.22494601](https://doi.org/10.6084/m9.figshare.22494601)

### Files

| File | Rows | Description |
|---|---|---|
| `clinicalnlp_taskB_test1.csv` | 40 | Primary data file. Contains four columns: `dataset` (subset name), `encounter_id` (unique identifier), `dialogue` (full doctor-patient conversation transcript), and `note` (gold standard clinical note used as the evaluation reference — automatically generated then reviewed and corrected by domain experts including medical scribes and physicians) |
| `clinicalnlp_taskB_test1_metadata.csv` | 40 | Metadata file. Contains: `dataset`, `encounter_id`, `id`, `doctor_name`, `patient_gender`, `patient_age`, `patient_firstname`, `patient_familyname`, `cc` (chief complaint), `2nd_complaints` (secondary complaints). Joined to the primary file on `encounter_id` |
| `clinicalnlp_taskB_test1_n5.csv` | 5 | First 5 encounters from the primary file — used for quick batch runs |
| `clinicalnlp_taskB_test1_metadata_n5.csv` | 5 | Metadata for the first 5 encounters — used alongside the n5 data file |

### Dataset subsets

The 40 encounters span three subsets, each representing a different mode of clinical note generation:

| Subset | Count | Description |
|---|---|---|
| `virtassist` | 10 | Doctor uses explicit wake words to activate a virtual assistant |
| `virtscribe` | 8 | Doctor directs a separate scribe; includes pre-ambles and after-visit dictations |
| `aci` | 22 | Natural doctor-patient conversation with no explicit assistant or scribe |

### Download instructions

The data files are already included in the repo for ease, but to download:
1. Go to [https://doi.org/10.6084/m9.figshare.22494601](https://doi.org/10.6084/m9.figshare.22494601)
2. Download the ACI-Bench release
3. Copy these two files into this `data/` directory:
   - `clinicalnlp_taskB_test1.csv`
   - `clinicalnlp_taskB_test1_metadata.csv`
4. We generated the n5 files to create a small test subset

### Citation

```bibtex
@article{aci-bench,
  author = {Wen{-}wai Yim and
            Yujuan Fu and
            Asma {Ben Abacha} and
            Neal Snider and Thomas Lin and Meliha Yetisgen},
  title = {ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
  journal = {Nature Scientific Data},
  year = {2023}
}
```
---

## ACI-Bench Baseline

Published baseline prediction outputs from the original ACI-Bench paper
([Yim et al. 2023](https://github.com/wyim/aci-bench)), used as comparison
points for our three techniques. All files in this folder are directly from [https://github.com/wyim/aci-bench](https://github.com/wyim/aci-bench),
none were created by us. 


All four CSVs cover the same 40 encounters from `clinicalnlp_taskB_test1.csv`
and use the four-column format expected by `evaluate_fullnote.py`:

| Column | Description |
|---|---|
| `Dialogues` | The original doctor-patient conversation transcript |
| `Reference Summaries` | The gold standard clinical note used for evaluation |
| `note` | The model's predicted/generated clinical note |
| `encounter_id` | Unique encounter identifier, e.g. `D2N088` |

### How to score a baseline

Must be run from the **project root** (`PythonProject4/`), not from inside `baselines/`:

```bash
cd ~/PycharmProjects/PythonProject4

python evaluation/evaluate_fullnote.py \
  data/clinicalnlp_taskB_test1.csv \
  baselines/predictions/aci_bench_paper/test1ChatGPT_.csv \
  data/clinicalnlp_taskB_test1_metadata.csv
```

Results are written to `results/test1ChatGPT_.json`.

### Reference scores (Yim et al. 2023, Table 6 — test set 1)

All scores are reported as F1 × 100.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | MEDCON |
|---|---|---|---|---|
| ChatGPT (gpt-3.5-turbo) | 47.44 | 19.01 | 42.47 | 55.84 |
| GPT-4 | 51.76 | 22.58 | 45.97 | 57.78 |
| text-davinci-002 | 41.08 | 17.27 | 37.46 | 47.39 |
| text-davinci-003 | 47.07 | 22.08 | 43.11 | 57.16 |

Our GVR pipeline (Llama-3.1-8B) on the same test set: ROUGE-1: 0.4082,
ROUGE-2: 0.1801, ROUGE-L: 0.2675. Note the paper reports scores × 100;
our evaluation script reports raw fractions.

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

## Running an Individual Technique

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

## How to Implement A New Technique

Your technique file only needs to do two things:

**1. Subclass `SOAPPipeline` and implement `run_pipeline()`**

```python
from experiments.shared_pipeline_elements.pipeline_base import SOAPPipeline, PipelineResult
from experiments.shared_pipeline_elements.aci_data_loader import ACIEncounter
from experiments.shared_pipeline_elements.pydantic_schema import SOAPNote


class MyTechniquePipeline(SOAPPipeline):
    TECHNIQUE_NAME = "my_technique"  # "prompt_engineering" or "constrained_decoding"

    def run_pipeline(self, encounter: ACIEncounter, max_retries: int = 0) -> PipelineResult:
        try:
            raw = self.call_llm(your_prompt)
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
    from experiments.shared_pipeline_elements.batch_runner import main

    main(
        MyTechniquePipeline(),
        default_results_path="results/my_results.json",
        default_max_retries=0,
    )
```

That's it. The batch loop, incremental saving, predictions CSV, and summary stats are all handled automatically.

---

## The Shared SOAPNote Schema

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


## Evaluation
Evaluation scripts for scoring generated SOAP notes against gold standard references.
All four files are adapted from the ACI-Bench repository (Yim et al. 2023) and are
published under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

### Files

| File | Description |
|---|---|
| `evaluate_fullnote.py` | Main evaluation script. Computes ROUGE, MEDCON (UMLS), BERTScore, and BLEURT across the full note and each of the four SOAP divisions. Outputs a JSON results file. |
| `UMLS_evaluation.py` | MEDCON scoring logic. Uses QuickUMLS to extract UMLS medical concepts from generated and reference notes, then computes F1 over matched concepts. Called by `evaluate_fullnote.py`. |
| `sectiontagger.py` | Rule-based section parser. Detects and splits a clinical note into its four divisions (`subjective`, `objective_exam`, `objective_results`, `assessment_and_plan`) using regex patterns. Called by `evaluate_fullnote.py`. |
| `semantics.py` | Loads the list of UMLS semantic type IDs used to filter MEDCON concept extraction to clinically relevant categories (Anatomy, Chemicals & Drugs, Disorders, etc.). Called by `UMLS_evaluation.py`. Reads from `resources/semantic_types.txt`. |

### Dependencies

| Script | Requires |
|---|---|
| `evaluate_fullnote.py` | `evaluate`, `pandas`, `numpy` |
| `UMLS_evaluation.py` | `quickumls`, `spacy`, `en_ner_bc5cdr_md` model, QuickUMLS index at `resources/des/` |
| `semantics.py` | `resources/semantic_types.txt` |

All UMLS/MEDCON-related scripts require the `umls_env` conda environment (Python 3.11).
The main `.venv` (Python 3.13) is **incompatible** with QuickUMLS due to a `leveldb` dependency.

### How to run

Must be run from the **project root** (`PythonProject4/`) with `umls_env` activated:

```bash
conda activate umls_env

python evaluation/evaluate_fullnote.py \
    data/clinicalnlp_taskB_test1.csv \
    results/gvr_results_predictions.csv \
    data/clinicalnlp_taskB_test1_metadata.csv
```

Arguments:
- `<gold>` — path to the gold CSV (`clinicalnlp_taskB_test1.csv`)
- `<sys>` — path to the predictions CSV (e.g. `results/gvr_results_predictions.csv`)
- `<metadata-file>` — path to the metadata CSV (`clinicalnlp_taskB_test1_metadata.csv`)

Output is written to `results/<predictions_filename>.json`.

### Metrics

| Metric | Status | Notes |
|---|---|---|
| ROUGE-1, ROUGE-2, ROUGE-Lsum | ✅ Working | `rougeLsum` is the summary-level variant used in the ACI-Bench paper |
| MEDCON (UMLS) | ✅ Working | Requires `umls_env` and QuickUMLS index |
| BERTScore | ⚠️ Stubbed | Returns 0.0 — pending Python 3.13/DeBERTa tokenizer compatibility fix |
| BLEURT | ⚠️ Stubbed | Returns 0.0 — pending Python 3.13/DeBERTa tokenizer compatibility fix |

### MEDCON - UMLS / QuickUMLS setup

MEDCON requires QuickUMLS, which depends on a C extension (`leveldb`) that only compiles correctly on **Python 3.11**. It will fail on the main `.venv` (Python 3.13) and on `acidemo` (Python 3.9). You need a dedicated environment.

This is a one-time setup per machine.

**1. Get the UMLS 2022AA metathesaurus**

You need a UMLS license. Request one at https://uts.nlm.nih.gov/uts/ if you don't have one.

Download the 2022AA release from:
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html

Extract the download and copy these three files into the `resources/` directory:
```
resources/
├── MRCONSO.RRF
├── MRSTY.RRF
└── semantic_types.txt        ← already in the repo
```

> These files are large and are not committed to the repo. Each person must download them individually.

**2. Create the QuickUMLS conda environment**

```bash
conda create -n umls_env python=3.11
conda activate umls_env
pip install quickumls
```

**3. Build the QuickUMLS index (one-time, takes several minutes)**

Run this from the project root using absolute paths:

```bash
python -m quickumls.install /absolute/path/to/resources/ /absolute/path/to/resources/des
```

For example:
```bash
python -m quickumls.install ~/PycharmProjects/PythonProject4/resources/ ~/PycharmProjects/PythonProject4/resources/des
```

When it finishes, `resources/des/` will exist. You never need to run this again.

**4. Run MEDCON evaluation**

Always switch to `umls_env` before running anything that imports QuickUMLS:

```bash
conda activate umls_env
python evaluation/UMLS_evaluation.py ...
```

> If you see `ModuleNotFoundError: No module named 'quickumls'`, you are in the wrong environment. Run `conda activate umls_env` first.

---

## Note on Constrained Decoding

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
