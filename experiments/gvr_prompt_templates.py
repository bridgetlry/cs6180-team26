from typing import get_args, get_origin, Union
from pydantic import ValidationError
from experiments.shared_pipeline_elements.pydantic_schema import SOAPNote


# ─────────────────────────────────────────────
# AUTO-GENERATED COMPACT SCHEMA
# Derived directly from SOAPNote.model_fields
# so prompt and schema are always in sync.
# ─────────────────────────────────────────────

def _get_type_hint(field_name: str, field_info) -> str:
    """Extract a human-readable type string from a Pydantic field annotation."""
    annotation = field_info.annotation

    # Unwrap Optional[X] -> X
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        annotation = args[0] if args else annotation

    type_map = {
        int: "integer",
        float: "float",
        str: "string",
        bool: "boolean",
    }
    base = type_map.get(annotation, str(annotation))

    # All optional fields can be null
    if field_info.default is None:
        return f"{base} or null"
    return base


def build_compact_schema() -> str:
    """
    Auto-generate a compact schema string from SOAPNote.model_fields.
    Includes working_space and encounter_id as prompt-only fields
    (stripped by extra='ignore' before Pydantic validation).
    Always in sync with the actual SOAPNote model.
    """
    lines = []

    # Prompt-only fields — not in SOAPNote but useful for model reasoning
    lines.append('  "working_space": "string (required) — reasoning scratchpad. Free text, NOT JSON. Write this first."')
    lines.append('  "encounter_id": "string (required) — copy the encounter ID exactly as provided"')
    lines.append("")

    # Required fields first
    required = []
    optional = []
    for field_name, field_info in SOAPNote.model_fields.items():
        is_optional = field_info.default is None
        if is_optional:
            optional.append((field_name, field_info))
        else:
            required.append((field_name, field_info))

    lines.append("  // --- Required fields ---")
    for field_name, field_info in required:
        type_hint = _get_type_hint(field_name, field_info)
        description = field_info.description or ""
        lines.append(f'  "{field_name}": "{type_hint} — {description}"')

    lines.append("")
    lines.append("  // --- Optional fields (use JSON null if not present in transcript) ---")
    for field_name, field_info in optional:
        type_hint = _get_type_hint(field_name, field_info)
        description = field_info.description or ""
        lines.append(f'  "{field_name}": "{type_hint} — {description}"')

    return "{\n" + ",\n".join(lines) + "\n}"


# Build once at import time — reused across all prompt calls
COMPACT_SCHEMA = build_compact_schema()


# ─────────────────────────────────────────────
# INITIAL PROMPT
# ─────────────────────────────────────────────
def build_initial_prompt(transcript: str, encounter_id: str) -> str:
    return f"""Task:
Read a doctor-patient conversation transcript and extract a structured SOAP note as a JSON object.

Role:
You are an experienced clinical documentation specialist. You perform this task with attention to detail, completeness, and accuracy.

Working Space — use the 'working_space' field FIRST before extracting anything:
- Read the entire transcript carefully from beginning to end
- Identify the chief complaint and overall clinical context
- Map content to each section: what belongs in subjective, objective_exam, objective_results, and assessment_and_plan
- Collect exact values for any typed fields: pain scores, age, vitals, substance use
- Note whether any lab or imaging results are discussed (objective_results)
- working_space must be free text, concise, NOT JSON format

Systematic approach (reason in working_space before producing JSON):
1. Read the full transcript start to finish — do not stop early
2. Identify content for each of the four required text divisions
3. Collect exact numeric values for typed fields only if explicitly stated
4. Determine urgency: only true if explicit emergency language present
5. Only then produce the final JSON

Rules:
1. Extract only information present in the transcript — do not fabricate
2. Required text fields (subjective, objective_exam, assessment_and_plan) must not be empty
3. objective_results: use empty string "" if no labs or imaging are discussed
4. Typed numeric fields: extract only if explicitly stated as a number in the transcript
5. Boolean fields (smoker, alcohol_use, recreational_drugs): true if explicitly stated, false if explicitly denied, null if not mentioned
6. is_urgent: null if unclear, true only for explicit emergency language
7. pain_severity: only if a numeric 0-10 score is explicitly stated — do not convert qualitative descriptions
8. temperature_f: in Fahrenheit only — convert from Celsius if needed (C * 9/5 + 32)
9. All optional fields not mentioned in transcript: use JSON null (not the string "null", not "N/A")
10. Output exactly one JSON object — no preamble, no markdown, no code blocks
11. Do not repeat any keys

SCHEMA:
{COMPACT_SCHEMA}

ENCOUNTER ID: {encounter_id}

TRANSCRIPT:
{transcript}

Reflection — before producing your final JSON, verify:
1. Is every required field populated and non-empty?
2. Are the four text divisions (subjective, objective_exam, objective_results, assessment_and_plan) correctly separated?
3. Is objective_results empty string if no labs/imaging were discussed?
4. Is pain_severity only set if a numeric score was explicitly stated?
5. Is is_urgent null unless explicit emergency language was used?
6. Are all null fields genuinely absent from the transcript?
7. Have you read the complete transcript including the end where plans are often stated?

JSON:"""


# ─────────────────────────────────────────────
# RETRY PROMPT
# ─────────────────────────────────────────────
def build_retry_prompt(
    transcript: str,
    encounter_id: str,
    previous_output: str,
    error_message: str,
) -> str:
    return f"""Task:
Your previous SOAP note extraction produced an invalid output. Fix it.

Role:
You are an experienced clinical documentation specialist. You perform this task with attention to detail, completeness, and accuracy.

Working Space — use 'working_space' to reason through the fix:
- Identify exactly which fields failed and why
- Review the relevant parts of the transcript for those fields only
- Confirm your corrected values before committing
- working_space must be free text, concise, NOT JSON format

Rules:
- Return ONLY a corrected valid JSON object
- Fix ONLY the fields mentioned in the validation error
- Do not change fields that were correct
- Do not fabricate information not present in the transcript
- Do not wrap in code blocks, no preamble, no markdown
- Do not repeat any keys
- For optional fields not present in the transcript: use JSON null (not the string "null", not "N/A")
- objective_results: use empty string "" if no labs or imaging discussed — not null

SCHEMA:
{COMPACT_SCHEMA}

ENCOUNTER ID: {encounter_id}

TRANSCRIPT:
{transcript}

YOUR PREVIOUS OUTPUT:
{previous_output}

VALIDATION ERROR:
{error_message}

Reflection — before producing corrected JSON:
1. Have you fixed exactly the fields mentioned in the error?
2. Have you left correct fields unchanged?
3. Does your fix introduce any new errors?

JSON:"""


# ─────────────────────────────────────────────
# VALIDATION ERROR FORMATTER
# ─────────────────────────────────────────────
def format_validation_error(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        field = err["loc"][0] if err["loc"] else "unknown"
        msg = err["msg"]
        lines.append(f"- Field '{field}': {msg}")
    return "The following fields failed validation:\n" + "\n".join(lines)


# ─────────────────────────────────────────────
# CLI — inspect generated schema
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Auto-generated compact schema ===")
    print(COMPACT_SCHEMA)
    print(f"\nSOAPNote has {len(SOAPNote.model_fields)} fields")
    required = [k for k, v in SOAPNote.model_fields.items() if v.default is None.__class__ or v.is_required()]
    print(f"Required: {[k for k, v in SOAPNote.model_fields.items() if v.is_required()]}")
