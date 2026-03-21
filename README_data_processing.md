# SOAP Data Processing Starter

This starter script prepares the transcript PDF for the rest of our CS6180 project.

## What it does
- extracts text from the PDF
- splits the dataset into scenarios using `[Scenario X]` and `[SEP]`
- parses doctor/patient turns (`D:` / `P:`)
- normalizes each transcript for downstream prompting
- exports JSONL, CSV, and SQLite versions of the dataset
- writes a Pydantic-style SOAP schema file for downstream structured generation

## Expected outputs
- `extracted_full_text.txt`
- `transcripts.jsonl`
- `scenarios.csv`
- `turns.csv`
- `soap_dataset.db`
- `soap_schema.json`
- `prompt_ready_inputs.json`
- `dataset_summary.json`

## Run
```bash
pip install pypdf pydantic
python soap_data_processing.py   --pdf combined_conversations_with_scenarios.pdf   --outdir processed_data
```
