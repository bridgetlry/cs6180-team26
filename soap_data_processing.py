from __future__ import annotations # for type hints to reference classes defined later in the file without needing string literals

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

# Uses Pydantic for structured data models and validation if available
# Falls back to simple dataclasses where Pydantic may not be installed
try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    Field = lambda default=None, **kwargs: default  # type: ignore


''' Schema definitions '''

# Used for structured representation of each speaker turn in the provided patient-doctor transcript
class TurnModel(BaseModel):
    turn_index: int
    speaker: str = Field(description="D for doctor, P for patient")
    text: str

# Represents a single patient-doctor scenario with its associated transcript and metadata
class TranscriptExample(BaseModel):
    scenario_id: int
    chief_complaint_hint: Optional[str] = None
    raw_transcript: str
    normalized_transcript: str
    turns: list[TurnModel]

# Target schema for the SOAP note generation output
class SOAPNote(BaseModel):
    # Minimal SQL-friendly schema for generation outputs.
    scenario_id: int
    subjective: str = Field(description="History, symptoms, PMH, medications, allergies, family/social history")
    objective: str = Field(description="Only explicitly observed or stated findings from the transcript")
    assessment: str = Field(description="Likely clinical impression or differential, grounded in transcript only")
    plan: str = Field(description="Next steps, tests, treatments, follow-up, grounded in transcript only")
    probable_diagnosis: Optional[str] = None
    severity: Optional[str] = None
    disposition: Optional[str] = None

@dataclass
# Structured representation of a single speaker turn
class Turn:
    turn_index: int
    speaker: str
    text: str


@dataclass
# Structured representation of a patient-doctor scenario with its transcript and metadata
class Scenario:
    scenario_id: int
    raw_transcript: str
    normalized_transcript: str
    chief_complaint_hint: Optional[str]
    turns: list[Turn]


''' PDF/text extraction '''

# Text extraction using pypdf as a fallback when llama_index's SimpleDirectoryReader is not available or fails
def extract_text_with_pypdf(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    texts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)

# Primary text extraction function that tries llama_index first and falls back to pypdf if needed
def extract_text_with_llama_index(pdf_path: Path) -> str:
    try:
        from llama_index.core import SimpleDirectoryReader

        docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
        if docs:
            return "\n".join(doc.text for doc in docs if getattr(doc, "text", None))
    except Exception:
        pass
    return extract_text_with_pypdf(pdf_path)


''' Cleaning + parsing helpers '''

# Normalizes whitespace and line breaks to create a cleaner base for parsing
def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Performs lightweight cleanup of common OCR errors observed in the transcripts - preserves the original text as much as possible for downstream processing and analysis
def cleanup_common_ocr_noise(text: str) -> str:
    replacements = {
        "D;": "D:",
        "D .": "D:",
        "P;": "P:",
        "[ SE P ]": "[SEP]",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Normalizes broken speaker prefixes at line starts
    text = re.sub(r"(?m)^([DP])\s*;", r"\1:", text)
    text = re.sub(r"(?m)^([DP])\s*[\.]", r"\1:", text)
    return text

# Looks for the chief (main) complaint or reason for visit in the patient's first statement using simple keyword patterns
# If no clear pattern is found, returns first 80 characacters of the first patient line as a hint for the chief complaint
# If no patient lines are found, returns None
def infer_chief_complaint(normalized_transcript: str) -> Optional[str]:
    patient_lines = [
        m.group(1).strip()
        for m in re.finditer(r"(?m)^P:\s*(.+)$", normalized_transcript)
    ]
    if not patient_lines:
        return None

    first = patient_lines[0].lower()
    patterns = [
        (r"chest pain", "chest pain"),
        (r"breathless|short of breath|difficulty breathing", "shortness of breath"),
        (r"rash", "rash"),
        (r"nausea|nauseated|vomit", "nausea/vomiting"),
        (r"abdomen|belly|abdominal pain", "abdominal pain"),
        (r"back pain|my back", "back pain"),
        (r"bladder|burning when i go|uti|urinary", "urinary symptoms"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, first):
            return label
    return first[:80]

# Parses cleaned scenario text into structured turns ensuring statements are correctly attributed to either doctor (D) or patient (P)
def parse_turns(scenario_body: str) -> list[Turn]:
    cleaned = scenario_body.strip()
    # Ensures speaker tags begin on their own lines for easier parsing
    cleaned = re.sub(r"(?<!\n)(?=(?:D:|P:))", "\n", cleaned)
    cleaned = re.sub(r"\n+", "\n", cleaned).strip()

    # Captures each speaker block until the next speaker or end
    pattern = re.compile(r"(?ms)^([DP]):\s*(.*?)\s*(?=^[DP]:|\Z)")
    turns: list[Turn] = []
    for idx, match in enumerate(pattern.finditer(cleaned), start=1):
        speaker, text = match.groups()
        text = normalize_whitespace(text)
        if text:
            turns.append(Turn(turn_index=idx, speaker=speaker, text=text))
    return turns

# Formats the list of structured turns back into a clean, normalized transcript string for easier readability and downstream processing
def format_normalized_transcript(turns: Iterable[Turn]) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)

# Main parsing function
# Takes full extracted text from the PDF, cleans it, and uses regex patterns to id and extract individual scenarios along with their transcripts
def parse_scenarios(full_text: str) -> list[Scenario]:
    text = normalize_whitespace(cleanup_common_ocr_noise(full_text))

    # Uses regex to id scenario blocks starting with [Scenario X] until end of next scenario or end of text
    scenario_pattern = re.compile(
        r"\[Scenario\s+(\d+)\]\s*(.*?)(?=(?:\[SEP\]\s*\[Scenario\s+\d+\])|\Z)",
        re.S,
    )

    # Iterates through all identified scenario blocks and extracts key info
    scenarios: list[Scenario] = []
    for match in scenario_pattern.finditer(text):
        scenario_id = int(match.group(1))
        body = match.group(2).strip()
        body = re.sub(r"\[SEP\]\s*$", "", body).strip()
        turns = parse_turns(body)
        normalized = format_normalized_transcript(turns)
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                raw_transcript=body,
                normalized_transcript=normalized,
                chief_complaint_hint=infer_chief_complaint(normalized),
                turns=turns,
            )
        )

    if not scenarios:
        raise ValueError(
            "No scenarios were parsed. Check the PDF text extraction or scenario regex."
        )
    return scenarios


''' Serialization / exports '''

# Converts structured Scenario objects into JSON records suitable for line-delimited JSON output
# Ensures all relevant info is included while keeping the structure flat for easier downstream processing and analysis
def scenario_to_json_record(s: Scenario) -> dict:
    return {
        "scenario_id": s.scenario_id,
        "chief_complaint_hint": s.chief_complaint_hint,
        "raw_transcript": s.raw_transcript,
        "normalized_transcript": s.normalized_transcript,
        "turns": [asdict(t) for t in s.turns],
        "num_turns": len(s.turns),
    }

# Writes a list of JSON records to a line-delimited JSONL file, ensuring UTF-8 encoding and proper handling of special characters
def write_jsonl(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Writes a JSON file containing the schemas for both the input transcripts and the target SOAP note outputs
# Uses Pydantic's model_json_schema if available for better documentation and validation of the expected data structures
def write_schema_file(output_path: Path) -> None:
    schema = {
        "input_record": TranscriptExample.model_json_schema() if hasattr(TranscriptExample, "model_json_schema") else {},
        "target_output": SOAPNote.model_json_schema() if hasattr(SOAPNote, "model_json_schema") else {},
    }
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

# Writes the structured scenarios and their turns into a SQLite database with two tables:
# 1. scenario-level info
# 2. individual turns
def write_sqlite_db(scenarios: list[Scenario], db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS scenarios;
        DROP TABLE IF EXISTS turns;

        CREATE TABLE scenarios (
            scenario_id INTEGER PRIMARY KEY,
            chief_complaint_hint TEXT,
            raw_transcript TEXT NOT NULL,
            normalized_transcript TEXT NOT NULL,
            num_turns INTEGER NOT NULL
        );

        CREATE TABLE turns (
            scenario_id INTEGER NOT NULL,
            turn_index INTEGER NOT NULL,
            speaker TEXT NOT NULL CHECK (speaker IN ('D', 'P')),
            utterance TEXT NOT NULL,
            PRIMARY KEY (scenario_id, turn_index),
            FOREIGN KEY (scenario_id) REFERENCES scenarios (scenario_id)
        );
        """
    )

    cur.executemany(
        """
        INSERT INTO scenarios (scenario_id, chief_complaint_hint, raw_transcript, normalized_transcript, num_turns)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                s.scenario_id,
                s.chief_complaint_hint,
                s.raw_transcript,
                s.normalized_transcript,
                len(s.turns),
            )
            for s in scenarios
        ],
    )

    cur.executemany(
        """
        INSERT INTO turns (scenario_id, turn_index, speaker, utterance)
        VALUES (?, ?, ?, ?)
        """,
        [
            (s.scenario_id, t.turn_index, t.speaker, t.text)
            for s in scenarios
            for t in s.turns
        ],
    )

    conn.commit()
    conn.close()

# Writes two CSV files:
# 1. scenarios.csv with one row per scenario containing scenario-level info and the full transcripts
# 2. turns.csv with one row per turn containing the scenario_id, turn index, speaker, and utterance text for easier analysis of dialogue structure and content at the turn level
def write_csv_files(scenarios: list[Scenario], out_dir: Path) -> None:
    import csv

    scenarios_csv = out_dir / "scenarios.csv"
    turns_csv = out_dir / "turns.csv"

    with scenarios_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_id",
                "chief_complaint_hint",
                "num_turns",
                "raw_transcript",
                "normalized_transcript",
            ],
        )
        writer.writeheader()
        for s in scenarios:
            writer.writerow(
                {
                    "scenario_id": s.scenario_id,
                    "chief_complaint_hint": s.chief_complaint_hint,
                    "num_turns": len(s.turns),
                    "raw_transcript": s.raw_transcript,
                    "normalized_transcript": s.normalized_transcript,
                }
            )

    with turns_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario_id", "turn_index", "speaker", "utterance"],
        )
        writer.writeheader()
        for s in scenarios:
            for t in s.turns:
                writer.writerow(
                    {
                        "scenario_id": s.scenario_id,
                        "turn_index": t.turn_index,
                        "speaker": t.speaker,
                        "utterance": t.text,
                    }
                )


# Writes a JSON file containing a list of records formatted for direct use as inputs to prompt-based generation of SOAP notes
# Each record includes the scenario ID, chief complaint hint, normalized transcript, and a clear instruction 
def write_prompt_ready_json(scenarios: list[Scenario], out_path: Path) -> None:
    payload = []
    for s in scenarios:
        payload.append(
            {
                "scenario_id": s.scenario_id,
                "chief_complaint_hint": s.chief_complaint_hint,
                "transcript": s.normalized_transcript,
                # Clear and specific instructions to guide the model in generating SOAP notes strictly based on provided transcript info
                "instruction": (
                    "Convert the patient-doctor transcript into a SQL-compatible SOAP note "
                    "that follows the target schema exactly. Use only information stated in the transcript."
                ),
            }
        )
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# Main function to orchestrate the entire data processing pipeline:
# 1. Extracts text from the provided PDF using llama_index with a pypdf fallback
# 2. Parses the extracted text into structured Scenario objects with their associated turns
# 3. Serializes the structured data into multiple formats for different use cases:
#    - Full extracted text for reference
#    - Line-delimited JSONL for easy loading in Python and other languages
#    - CSV files for analysis and spreadsheet use
#    - SQLite database for structured querying and integration with other tools
#    - JSON schema file for documentation and validation of expected data structures
#    - Prompt-ready JSON for direct use in generation tasks     
def build_outputs(pdf_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    text = extract_text_with_llama_index(pdf_path)
    scenarios = parse_scenarios(text)
    json_records = [scenario_to_json_record(s) for s in scenarios]

    (output_dir / "extracted_full_text.txt").write_text(text, encoding="utf-8")
    write_jsonl(json_records, output_dir / "transcripts.jsonl")
    write_csv_files(scenarios, output_dir)
    write_sqlite_db(scenarios, output_dir / "soap_dataset.db")
    write_schema_file(output_dir / "soap_schema.json")
    write_prompt_ready_json(scenarios, output_dir / "prompt_ready_inputs.json")

    summary = {
        "num_scenarios": len(scenarios),
        "min_turns": min(len(s.turns) for s in scenarios),
        "max_turns": max(len(s.turns) for s in scenarios),
        "avg_turns": round(sum(len(s.turns) for s in scenarios) / len(scenarios), 2),
        "scenario_ids": [s.scenario_id for s in scenarios[:10]],
        "artifacts": [
            "extracted_full_text.txt",
            "transcripts.jsonl",
            "scenarios.csv",
            "turns.csv",
            "soap_dataset.db",
            "soap_schema.json",
            "prompt_ready_inputs.json",
        ],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


# CLI for running the data processing pipeline from the command line

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and structure patient-doctor transcripts from the project PDF."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to combined_conversations_with_scenarios.pdf",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("processed_data"),
        help="Directory for processed outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_outputs(args.pdf, args.outdir)
