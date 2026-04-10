from __future__ import annotations # for type hints to reference classes defined later in the file without needing string literals

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

import instructor
import sys
from openai import OpenAI
from dotenv import load_dotenv
import jsonlines
import os
from soap_data_processing import SOAPNote # using SOAPNote schema from dataprocessing

# Generates and prints a SOAP Note output for the given scenario (1-26)
def generate_notes(scenario_number):

    # client = instructor.from_provider("qwen/qwen-turbo") # from documentation

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = instructor.from_openai(
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    )

    scenario = ""
    with jsonlines.open("processed_data/transcripts.jsonl") as file:
        for line_num, line in enumerate(file):
            if line_num == scenario_number - 1:
                scenario = line
                break

    instruction = "Convert the patient-doctor transcript into a SQL-compatible SOAP note that follows the target schema exactly. Use only information stated in the transcript."

    prompt_text = f"{instruction} Transcript: {scenario['raw_transcript']} Hint: {scenario['chief_complaint_hint']}"

    soap_note = client.create(
        model="qwen/qwen-turbo",
        response_model=SOAPNote,
        messages=[
            {"role": "user", "content": prompt_text}
        ]
    )

    print(soap_note)


if __name__ == "__main__":

    scenario_arg = ""
    scenario_int = 0
    
    try:
        print("Starting constrained decoding...")
        try:
            scenario_arg = sys.argv[1]
            scenario_int = int(scenario_arg)
            if scenario_int < 1 or scenario_int > 26:
                raise Exception()
        except:
            print("Scenario number must be an integer between 1 and 26, inclusive")
            quit()

        generate_notes(scenario_int)
            
    except Exception as e :
        print(f"Unable to process: {e}")
        quit()





