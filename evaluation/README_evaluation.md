# evaluation/

Evaluation scripts for scoring generated SOAP notes against gold standard references.
All four files are adapted from the ACI-Bench repository (Yim et al. 2023) and are
published under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

---

## Files

| File | Description |
|---|---|
| `evaluate_fullnote.py` | Main evaluation script. Computes ROUGE, MEDCON (UMLS), BERTScore, and BLEURT across the full note and each of the four SOAP divisions. Outputs a JSON results file. |
| `UMLS_evaluation.py` | MEDCON scoring logic. Uses QuickUMLS to extract UMLS medical concepts from generated and reference notes, then computes F1 over matched concepts. Called by `evaluate_fullnote.py`. |
| `sectiontagger.py` | Rule-based section parser. Detects and splits a clinical note into its four divisions (`subjective`, `objective_exam`, `objective_results`, `assessment_and_plan`) using regex patterns. Called by `evaluate_fullnote.py`. |
| `semantics.py` | Loads the list of UMLS semantic type IDs used to filter MEDCON concept extraction to clinically relevant categories (Anatomy, Chemicals & Drugs, Disorders, etc.). Called by `UMLS_evaluation.py`. Reads from `resources/semantic_types.txt`. |

---

## Dependencies

| Script | Requires |
|---|---|
| `evaluate_fullnote.py` | `evaluate`, `pandas`, `numpy` |
| `UMLS_evaluation.py` | `quickumls`, `spacy`, `en_ner_bc5cdr_md` model, QuickUMLS index at `resources/des/` |
| `semantics.py` | `resources/semantic_types.txt` |

All UMLS/MEDCON-related scripts require the `umls_env` conda environment (Python 3.11).
The main `.venv` (Python 3.13) is **incompatible** with QuickUMLS due to a `leveldb` dependency.

---

## How to run

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

---

## Metrics

| Metric | Status | Notes |
|---|---|---|
| ROUGE-1, ROUGE-2, ROUGE-Lsum | ✅ Working | `rougeLsum` is the summary-level variant used in the ACI-Bench paper |
| MEDCON (UMLS) | ✅ Working | Requires `umls_env` and QuickUMLS index |
| BERTScore | ⚠️ Stubbed | Returns 0.0 — pending Python 3.13/DeBERTa tokenizer compatibility fix |
| BLEURT | ⚠️ Stubbed | Returns 0.0 — pending Python 3.13/DeBERTa tokenizer compatibility fix |

---

## UMLS / QuickUMLS setup

See the root `README.md` for full setup instructions. In brief:

1. Download UMLS 2022AA (`MRCONSO.RRF`, `MRSTY.RRF`) and place in `resources/`
2. Build the QuickUMLS index: `python -m quickumls.install resources/ resources/des`
3. Always activate `umls_env` before running any evaluation

---

## Source

Yim et al. (2023). *ACI-Bench: a Novel Ambient Clinical Intelligence Dataset
for Benchmarking Automatic Visit Note Generation.* Scientific Data 10, 586.
https://doi.org/10.1038/s41597-023-02487-3
