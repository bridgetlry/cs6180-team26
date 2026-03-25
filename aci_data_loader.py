import json
import csv
import os
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────
# Primary files — use these for your main experiments
TRAIN_JSON      = "aci_bench_data/src_experiment_data_json/train_aci_asrcorr.json"
TEST_JSON       = "aci_bench_data/src_experiment_data_json/test1_aci_asr.json"
VALID_JSON      = "aci_bench_data/src_experiment_data_json/valid_aci_asr.json"

# Optional metadata CSVs — for chief complaint ground truth
TRAIN_META_CSV  = "aci_bench_data/src_experiment_data/train_aci_asrcorr_metadata.csv"
TEST_META_CSV   = "aci_bench_data/src_experiment_data/test1_aci_asr_metadata.csv"
VALID_META_CSV  = "aci_bench_data/src_experiment_data/valid_aci_asr_metadata.csv"


# ─────────────────────────────────────────────
# ENCOUNTER DATACLASS
# ─────────────────────────────────────────────
@dataclass
class ACIEncounter:
    encounter_id: str               # from 'file' field e.g. "0-aci"
    transcript: str                 # 'src' — doctor-patient conversation
    reference_note: str             # 'tgt' — clinician-written SOAP note (ground truth)
    dataset: str                    # always 'aci' for these files
    # from metadata CSV (optional — None if metadata not loaded)
    chief_complaint_gt: Optional[str] = None   # ground truth cc from metadata
    secondary_complaints: Optional[str] = None
    gender: Optional[str] = None


# ─────────────────────────────────────────────
# METADATA LOADER
# ─────────────────────────────────────────────
def load_metadata(meta_csv_path: str) -> dict:
    """
    Load metadata CSV into a dict keyed by encounter_id.
    Returns empty dict if file not found.
    """
    metadata = {}
    if not os.path.exists(meta_csv_path):
        print(f"[warn] metadata file not found: {meta_csv_path}")
        return metadata
    with open(meta_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # strip whitespace from all keys and values
            row = {k.strip(): v.strip() for k, v in row.items()}
            eid = row.get("encounter_id", "")
            metadata[eid] = row
    return metadata


# ─────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────
def load_aci_json(
    json_path: str,
    meta_csv_path: Optional[str] = None
) -> list[ACIEncounter]:
    """
    Load an ACI-bench JSON file into a list of ACIEncounter objects.
    Optionally merges metadata CSV for chief complaint ground truth.

    JSON structure:
        {"data": [{"src": "...", "tgt": "...", "file": "0-aci"}, ...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    metadata = load_metadata(meta_csv_path) if meta_csv_path else {}

    encounters = []
    for item in raw["data"]:
        file_id  = str(item.get("file", ""))
        src      = str(item.get("src", "")).strip()
        tgt      = str(item.get("tgt", "")).strip()
        dataset  = file_id.split("-")[-1] if "-" in file_id else "aci"

        # look up metadata by file_id or by numeric part
        meta = metadata.get(file_id, {})

        enc = ACIEncounter(
            encounter_id        = file_id,
            transcript          = src,
            reference_note      = tgt,
            dataset             = dataset,
            chief_complaint_gt  = meta.get("cc") or None,
            secondary_complaints= meta.get("2nd_complaints") or None,
            gender              = meta.get("gender") or None,
        )
        encounters.append(enc)

    return encounters


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────
def load_train(with_metadata: bool = True) -> list[ACIEncounter]:
    return load_aci_json(
        TRAIN_JSON,
        TRAIN_META_CSV if with_metadata else None
    )

def load_test(with_metadata: bool = True) -> list[ACIEncounter]:
    return load_aci_json(
        TEST_JSON,
        TEST_META_CSV if with_metadata else None
    )

def load_valid(with_metadata: bool = True) -> list[ACIEncounter]:
    return load_aci_json(
        VALID_JSON,
        VALID_META_CSV if with_metadata else None
    )

def format_transcript(encounter: ACIEncounter) -> str:
    """
    Return transcript text ready to pass into a pipeline prompt.
    Preserves [doctor] / [patient] speaker tags.
    """
    return encounter.transcript

def parse_reference_sections(reference_note: str) -> dict:
    """
    Parse a clinician reference note into its SOAP sections.
    Uses section headers as delimiters.
    Returns dict with keys: chief_complaint, history, physical_exam,
    assessment, plan, results (all Optional[str]).

    Useful for field-level ground truth comparison.
    """
    import re
    sections = {}
    # common section headers in ACI-bench notes
    headers = [
        "CHIEF COMPLAINT",
        "HISTORY OF PRESENT ILLNESS",
        "MEDICAL HISTORY",
        "SURGICAL HISTORY",
        "SOCIAL HISTORY",
        "FAMILY HISTORY",
        "REVIEW OF SYSTEMS",
        "VITALS",
        "PHYSICAL EXAM",
        "RESULTS",
        "ASSESSMENT AND PLAN",
        "ASSESSMENT",
        "PLAN",
        "INSTRUCTIONS",
    ]
    # build regex to split on headers
    pattern = "(" + "|".join(re.escape(h) for h in headers) + ")"
    parts = re.split(pattern, reference_note)

    current_header = None
    for part in parts:
        part = part.strip()
        if part in headers:
            current_header = part
        elif current_header and part:
            sections[current_header] = part

    return sections


# ─────────────────────────────────────────────
# MAIN — preview the dataset
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading train split (aci subset, ASR-corrected)...")
    train = load_train()
    print(f"Train encounters: {len(train)}")

    print("\nLoading test split (aci subset, ASR)...")
    test = load_test()
    print(f"Test encounters: {len(test)}")

    print("\n--- First encounter preview ---")
    enc = train[0]
    print(f"ID:              {enc.encounter_id}")
    print(f"Dataset:         {enc.dataset}")
    print(f"Gender:          {enc.gender}")
    print(f"CC (metadata):   {enc.chief_complaint_gt}")

    print(f"\nTranscript preview (first 300 chars):")
    print(enc.transcript[:300])

    print(f"\nReference note preview (first 300 chars):")
    print(enc.reference_note[:300])

    print(f"\nParsed reference sections:")
    sections = parse_reference_sections(enc.reference_note)
    for k, v in sections.items():
        print(f"  [{k}]: {v[:80]}...")