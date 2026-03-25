from soap_schema import SOAPNote  # shared schema imported by all three techniques

# ─────────────────────────────────────────────
# COMPACT SCHEMA
# Shown in prompt so model knows exact output format
# ─────────────────────────────────────────────
COMPACT_SCHEMA = """
{
  "working_space": "string (required) — your reasoning before extracting",
  "encounter_id": "string (required) — the encounter ID provided",

  "subjective": "string (required) — patient narrative: symptoms, history, medications, allergies, social/family history",
  "objective": "string (required) — explicitly observed findings only: vitals, physical exam, reported measurements",
  "assessment": "string (required) — clinical impression or differential, grounded in transcript only",
  "plan": "string (required) — tests, treatments, medications, follow-up. Infer if not explicit and set plan_is_inferred to true",
  "plan_is_inferred": "boolean (required) — true if inferred, false if explicitly stated",

  "chief_complaint": "string (required) — primary reason for visit in one sentence",
  "is_urgent": "boolean (required) — true only if explicit urgency or emergency language present",

  "patient_age": "integer or null — only if explicitly stated as a number",
  "patient_gender": "string or null — only if explicitly mentioned",
  "pain_severity": "float 0.0-10.0 or null — only if a numeric score explicitly stated",
  "temperature": "float 35.0-42.0 Celsius or null — convert from Fahrenheit if needed",
  "smoker": "boolean or null — true if current smoker explicitly stated, null if not mentioned",
  "alcohol_use": "boolean or null — true if explicitly mentioned, null if not mentioned",
  "recreational_drugs": "boolean or null — true if explicitly mentioned, null if not mentioned",

  "symptom_duration": "string or null",
  "pain_quality": "string or null",
  "pain_location": "string or null",
  "associated_symptoms": "string or null",
  "past_medical_history": "string or null",
  "current_medications": "string or null",
  "allergies": "string or null",
  "family_history": "string or null",
  "occupation": "string or null",
  "probable_diagnosis": "string or null"
}
"""


# ─────────────────────────────────────────────
# INITIAL PROMPT
# ─────────────────────────────────────────────
def build_initial_prompt(transcript: str, encounter_id: str) -> str:
    return f"""Task:
Your task is to read a doctor-patient conversation transcript and extract a structured SOAP note using the JSON schema below.

Role:
You are an experienced clinical documentation specialist. You perform this task with attention to detail, completeness, and accuracy. Believe in yourself and try your best.

Working Space — use the 'working_space' field FIRST before extracting anything:
- Read the entire transcript carefully from beginning to end
- Identify the chief complaint and overall clinical context
- Track scattered information: symptoms mentioned early, history mentioned late, plan mentioned throughout
- Note any structured data present: pain scores (exact quotes), patient age, vitals, substance use
- Flag whether a plan is explicitly stated or must be inferred
- Note any urgency indicators
- Verify you have read the complete transcript before proceeding
- working_space must be free text, concise, NOT JSON format

Systematic approach (write in working_space before producing JSON):
1. Read the full transcript from beginning to end
2. Identify content for each SOAP section
3. Collect exact quotes of any structured data: pain scores, age, temperature, substance use
4. Determine whether the plan is explicit or needs to be inferred
5. Check for urgency language
6. Do NOT stop after finding initial content — continue through the full transcript
7. Only then produce the final JSON

Rules:
1. Extract only information present in the transcript — do not fabricate
2. For typed fields (patient_age, pain_severity, temperature): extract only if explicitly stated as a number
3. For boolean fields (smoker, alcohol_use, recreational_drugs): true only if explicitly stated, null if not mentioned, false only if explicitly denied
4. is_urgent: true only for explicit emergency language — severe distress, immediate attention, emergency. Default false if uncertain
5. pain_severity: only if a numeric 0-10 score is explicitly stated. Do not convert qualitative descriptions
6. Set optional fields to null if not mentioned — do not guess
7. Output exactly one JSON object per transcript
8. Do not repeat any keys
9. For ALL optional fields — if the value is not present in the transcript, output the JSON value null with no quotes. Never output the string "null", "N/A", "not mentioned", or any other placeholder. Use actual JSON null.

Output format:
- Return ONLY a valid JSON object matching the schema below
- No explanation, preamble, or markdown formatting
- Do not wrap in code blocks
- Do not repeat any keys

SCHEMA:
{COMPACT_SCHEMA}

ENCOUNTER ID: {encounter_id}

TRANSCRIPT:
{transcript}

Reflection — before producing your final JSON, check:
1. Is every required field populated?
2. Is chief_complaint grounded in what the patient explicitly said?
3. Is pain_severity only set if a numeric score was explicitly stated?
4. Is is_urgent based on explicit emergency language, not general concern?
5. Is plan_is_inferred set correctly?
6. Are all null fields genuinely absent from the transcript?
7. Have you read the complete transcript, including the end where plans are often stated?

Use your reflections to adjust your output if needed.

JSON:"""


# ─────────────────────────────────────────────
# RETRY PROMPT
# Same structure, feedback injected at bottom
# Mirrors Dao et al. feedback injection pattern
# ─────────────────────────────────────────────
def build_retry_prompt(
    transcript: str,
    encounter_id: str,
    previous_output: str,
    error_message: str
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
- Do not wrap in code blocks
- Do not repeat any keys
- For optional fields not present in the transcript, use actual JSON null — not the string "null", not "N/A", not "not mentioned"

SCHEMA:
{COMPACT_SCHEMA}

ENCOUNTER ID: {encounter_id}

TRANSCRIPT:
{transcript}

YOUR PREVIOUS OUTPUT:
{previous_output}

VALIDATION ERROR:
{error_message}

Reflection — before producing corrected JSON, check:
1. Have you fixed exactly the fields mentioned in the error?
2. Have you left correct fields unchanged?
3. Does your fix introduce any new errors?

Use your reflections to adjust your output if needed.

JSON:"""


# ─────────────────────────────────────────────
# VALIDATION ERROR FORMATTER
# Converts Pydantic ValidationError into
# human-readable feedback for retry prompt
# ─────────────────────────────────────────────
def format_validation_error(e) -> str:
    from pydantic import ValidationError
    lines = []
    for err in e.errors():
        field = err["loc"][0] if err["loc"] else "unknown"
        msg = err["msg"]
        lines.append(f"- Field '{field}': {msg}")
    return "The following fields failed validation:\n" + "\n".join(lines)