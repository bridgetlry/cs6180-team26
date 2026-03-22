# SOAP Notes Generator -- Constrained Decoding

## What it does
- imports `SOAPNote` schema and preprocessed JSONL
- utilizes `qwen/qwen-turbo` LLM to generate SOAP notes in proper format
- prints output (SOAP notes) of the scenario with the given id (1-26)

## Setup
In order to use the LLM, a `.env` file must exist with the value `OPENROUTER_API_KEY=<your-api-key>`

## Run
```bash
pip install pypdf pydantic instructor openai jsonlines
python constrained_decoding.py <scenario_id>
```