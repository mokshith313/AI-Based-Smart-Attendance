import logging
from typing import Dict, Any
from app.llm_client import LLMClient
from app.schemas import CandidateExtractionSchema

logger = logging.getLogger("smart_screener.extractor")

EXTRACTION_SYSTEM_PROMPT = """
You are an expert HR Data Parser. Your job is to analyze raw resume text and extract structured information in JSON format.
You must return ONLY a JSON object conforming strictly to the requested schema. Do not include markdown formatting or commentary outside the JSON object.
"""

EXTRACTION_USER_PROMPT = """
Extract structured candidate profile information from the following raw resume text.

Resume Text:
----------------------------------------
{raw_text}
----------------------------------------

Return a JSON object with the following fields:
- "candidate_name": Full name of the candidate (string)
- "contact_email": Candidate email address if present (string or null)
- "contact_phone": Candidate phone number if present (string or null)
- "skills": Array of technical, framework, tool, and professional skills (array of strings)
- "experience": Array of work experiences, each object containing:
    - "role": Job title / role (string)
    - "company": Company or organization name (string)
    - "duration": Duration of employment (string, e.g. "2021 - 2023" or "2 years")
    - "responsibilities": Summary of key duties and achievements (string)
- "education": Array of education entries, each object containing:
    - "degree": Degree name / qualification (string)
    - "institution": School, college, or university name (string)
    - "year": Graduation year or period (string)
"""

class ResumeExtractor:
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()

    def extract(self, raw_text: str) -> Dict[str, Any]:
        """
        Extracts structured skills, experience, and education from raw resume text.
        Handles missing sections gracefully.
        """
        if not raw_text or not raw_text.strip():
            return CandidateExtractionSchema(
                candidate_name="Empty Resume",
                skills=[],
                experience=[],
                education=[]
            ).model_dump()

        prompt = EXTRACTION_USER_PROMPT.format(raw_text=raw_text)

        try:
            raw_result = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT
            )
            
            # Validate output with Pydantic
            candidate_data = CandidateExtractionSchema(**raw_result)
            return candidate_data.model_dump()
        except Exception as e:
            logger.error(f"Error during structured extraction: {e}. Returning safe fallback schema.")
            # Perform resilient fallback
            fallback_res = self.llm_client._heuristic_fallback(prompt)
            return CandidateExtractionSchema(**fallback_res).model_dump()
