"""
aci_data_loader.py
------------------
Shared data loader for the ACI-Bench test1 dataset.
Used by all three techniques: prompt engineering, constrained decoding, and GVR.

Primary sources (from ACI-Bench figshare download):
  - clinicalnlp_taskB_test1.csv          columns: dataset, encounter_id, dialogue, note
  - clinicalnlp_taskB_test1_metadata.csv columns: dataset, encounter_id, id,
                                                   doctor_name, patient_gender, patient_age,
                                                   patient_firstname, patient_familyname,
                                                   cc, 2nd_complaints

Optional baseline file:
  - test1ChatGPT_.csv  -- ChatGPT baseline predictions (joined on encounter_id)

Usage:
    from aci_data_loader import load_test1_encounters, save_predictions, ACIEncounter

    encounters = load_test1_encounters(
        data_csv="data/clinicalnlp_taskB_test1.csv",
        metadata_csv="data/clinicalnlp_taskB_test1_metadata.csv",
    )
    for enc in encounters:
        print(enc.encounter_id, enc.transcript[:80])
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ACIEncounter:
    """One doctor-patient encounter from ACI-Bench test1."""
    # Core fields (from clinicalnlp_taskB_test1.csv)
    encounter_id: str       # e.g. "D2N088"
    dataset: str            # "aci", "virtassist", or "virtscribe"
    transcript: str         # full doctor-patient conversation
    reference_note: str     # gold human-written SOAP note

    # Metadata fields (from clinicalnlp_taskB_test1_metadata.csv)
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    patient_firstname: Optional[str] = None
    patient_familyname: Optional[str] = None
    chief_complaint: Optional[str] = None
    secondary_complaints: list = field(default_factory=list)
    doctor_name: Optional[str] = None

    # Optional: ChatGPT baseline prediction (from test1ChatGPT_.csv)
    chatgpt_note: Optional[str] = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_test1_encounters(
    data_csv,
    metadata_csv=None,
    chatgpt_csv=None,
):
    """
    Load all 40 ACI-Bench test1 encounters.

    Args:
        data_csv:     Path to clinicalnlp_taskB_test1.csv (required)
        metadata_csv: Path to clinicalnlp_taskB_test1_metadata.csv (optional but recommended)
        chatgpt_csv:  Path to test1ChatGPT_.csv (optional, for baseline comparison)

    Returns:
        List of ACIEncounter objects in original order (D2N088-D2N127).

    Raises:
        FileNotFoundError: if data_csv does not exist.
        ValueError: if required columns are missing.
    """
    data_csv = Path(data_csv)
    if not data_csv.exists():
        raise FileNotFoundError(
            f"ACI-Bench test1 CSV not found at: {data_csv}\n"
            f"Expected: clinicalnlp_taskB_test1.csv\n"
            f"Download from: https://doi.org/10.6084/m9.figshare.22494601"
        )

    # --- Load core data ---
    encounters = []
    with open(data_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"encounter_id", "dialogue", "note", "dataset"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"clinicalnlp_taskB_test1.csv missing columns: {missing}")

        for row in reader:
            encounters.append(ACIEncounter(
                encounter_id=row["encounter_id"].strip(),
                dataset=row["dataset"].strip(),
                transcript=row["dialogue"].strip(),
                reference_note=row["note"].strip(),
            ))

    # Index by encounter_id for joining
    enc_index = {enc.encounter_id: enc for enc in encounters}

    # --- Join metadata (optional) ---
    if metadata_csv is not None:
        metadata_csv = Path(metadata_csv)
        if not metadata_csv.exists():
            print(f"Warning: metadata CSV not found at {metadata_csv}, skipping.")
        else:
            with open(metadata_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eid = row["encounter_id"].strip()
                    if eid not in enc_index:
                        continue
                    enc = enc_index[eid]
                    enc.patient_age = row.get("patient_age", "").strip() or None
                    enc.patient_gender = row.get("patient_gender", "").strip() or None
                    enc.patient_firstname = row.get("patient_firstname", "").strip() or None
                    enc.patient_familyname = row.get("patient_familyname", "").strip() or None
                    enc.chief_complaint = row.get("cc", "").strip() or None
                    enc.doctor_name = row.get("doctor_name", "").strip() or None
                    raw_secondary = row.get("2nd_complaints", "").strip()
                    enc.secondary_complaints = (
                        [s.strip() for s in raw_secondary.split(";") if s.strip()]
                        if raw_secondary else []
                    )

    # --- Join ChatGPT baseline (optional) ---
    if chatgpt_csv is not None:
        chatgpt_csv = Path(chatgpt_csv)
        if not chatgpt_csv.exists():
            print(f"Warning: ChatGPT CSV not found at {chatgpt_csv}, skipping.")
        else:
            with open(chatgpt_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eid = row["encounter_id"].strip()
                    if eid in enc_index:
                        enc_index[eid].chatgpt_note = row["note"].strip()

    return encounters


def get_encounter_by_id(encounters, encounter_id):
    """
    Get a single encounter by ID from an already-loaded list.
    Useful for single-record testing during development.

    Example:
        encounters = load_test1_encounters(...)
        enc = get_encounter_by_id(encounters, "D2N088")
    """
    for enc in encounters:
        if enc.encounter_id == encounter_id:
            return enc
    return None


# ---------------------------------------------------------------------------
# Output helper -- matches paper four-column output format
# ---------------------------------------------------------------------------

def save_predictions(predictions, output_path, encounters=None):
    """
    Save generated notes to a CSV matching the paper output format.

    If encounters is provided, outputs four columns matching test1ChatGPT_.csv:
        Dialogues, Reference Summaries, note, encounter_id

    If encounters is not provided, outputs two columns (minimal format):
        encounter_id, note

    Args:
        predictions: Dict mapping encounter_id -> generated note text.
        output_path: Where to write the output CSV.
        encounters:  List of ACIEncounter objects (optional but recommended).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if encounters is not None:
        enc_index = {enc.encounter_id: enc for enc in encounters}
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["Dialogues", "Reference Summaries", "note", "encounter_id"]
            )
            writer.writeheader()
            for encounter_id, note in predictions.items():
                enc = enc_index.get(encounter_id)
                writer.writerow({
                    "Dialogues": enc.transcript if enc else "",
                    "Reference Summaries": enc.reference_note if enc else "",
                    "note": note,
                    "encounter_id": encounter_id,
                })
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["encounter_id", "note"])
            writer.writeheader()
            for encounter_id, note in predictions.items():
                writer.writerow({"encounter_id": encounter_id, "note": note})

    print(f"Saved {len(predictions)} predictions to {output_path}")


# ---------------------------------------------------------------------------
# CLI -- quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from collections import Counter

    data_csv = sys.argv[1] if len(sys.argv) > 1 else "data/clinicalnlp_taskB_test1.csv"
    metadata_csv = sys.argv[2] if len(sys.argv) > 2 else "data/clinicalnlp_taskB_test1_metadata.csv"
    chatgpt_csv = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        encounters = load_test1_encounters(data_csv, metadata_csv, chatgpt_csv)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    datasets = Counter(enc.dataset for enc in encounters)
    print(f"Loaded {len(encounters)} encounters")
    print(f"Dataset breakdown: {dict(datasets)}")
    print(f"Encounter IDs: {encounters[0].encounter_id} ... {encounters[-1].encounter_id}")
    print()

    enc = encounters[0]
    print(f"=== {enc.encounter_id} ({enc.dataset}) ===")
    print(f"Patient:    {enc.patient_firstname} {enc.patient_familyname}, {enc.patient_age}yo {enc.patient_gender}")
    print(f"CC:         {enc.chief_complaint}")
    print(f"2nd:        {enc.secondary_complaints}")
    print(f"Transcript: {enc.transcript[:120]}...")
    print(f"Gold note:  {enc.reference_note[:120]}...")
    if enc.chatgpt_note:
        print(f"ChatGPT:    {enc.chatgpt_note[:120]}...")