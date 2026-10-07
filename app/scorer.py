import json
import logging
from typing import Dict, Any
from app.llm_client import LLMClient
from app.schemas import MatchScoreSchema

logger = logging.getLogger("smart_screener.scorer")

SCORING_SYSTEM_PROMPT = """
You are an expert HR Talent Evaluator & AI Recruiter. Your task is to perform an objective, evidence-based evaluation comparing a candidate's structured resume against a given Job Description.

You must output ONLY a valid JSON object matching the required schema. Do not include markdown preamble or commentary.
"""

SCORING_USER_PROMPT = """
Compare the following candidate resume profile with the job description. Evaluate fit on a scale of 1 to 10 (where 1 = completely unqualified, 10 = exceptional fit), and provide an evidence-backed justification grounded directly in the candidate's actual skills and experience.

Candidate Structured Profile:
----------------------------------------
Candidate Name: {candidate_name}
Skills: {skills}

Work Experience:
{experience_str}

Education:
{education_str}
----------------------------------------

Job Description:
----------------------------------------
{job_description}
----------------------------------------

Return a JSON object with the following fields:
- "score": Fit score as a number between 1.0 and 10.0 (e.g. 8.5)
- "justification": Detailed, human-readable justification referencing specific skills, projects, and experience from the candidate's resume that match or miss the job requirements (string)
- "pros": Array of key strengths and matching qualifications (array of strings)
- "cons": Array of gaps, missing experience, or areas of concern relative to the JD (array of strings)
- "matching_skills": Array of skills found in both candidate resume and job description (array of strings)
- "missing_skills": Array of required job description skills missing from the candidate's resume (array of strings)
"""

class ResumeScorer:
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()

    def score_candidate(self, candidate_data: Dict[str, Any], job_description_text: str) -> Dict[str, Any]:
        """
        Computes a 1-10 match score and justification comparing candidate profile against job description.
        """
        candidate_name = candidate_data.get("candidate_name", "Candidate")
        skills = candidate_data.get("skills", [])
        
        # Format experience list into human-readable string
        exp_list = candidate_data.get("experience", [])
        exp_lines = []
        for exp in exp_list:
            role = exp.get("role", "")
            company = exp.get("company", "")
            duration = exp.get("duration", "")
            resp = exp.get("responsibilities", "")
            exp_lines.append(f"- {role} at {company} ({duration}): {resp}")
        experience_str = "\n".join(exp_lines) if exp_lines else "None listed"

        # Format education list
        edu_list = candidate_data.get("education", [])
        edu_lines = []
        for edu in edu_list:
            degree = edu.get("degree", "")
            inst = edu.get("institution", "")
            year = edu.get("year", "")
            edu_lines.append(f"- {degree}, {inst} ({year})")
        education_str = "\n".join(edu_lines) if edu_lines else "None listed"

        prompt = SCORING_USER_PROMPT.format(
            candidate_name=candidate_name,
            skills=json.dumps(skills),
            experience_str=experience_str,
            education_str=education_str,
            job_description=job_description_text
        )

        try:
            raw_res = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=SCORING_SYSTEM_PROMPT
            )

            # Clamp score bounded between 1.0 and 10.0
            raw_score = float(raw_res.get("score", 5.0))
            clamped_score = min(10.0, max(1.0, round(raw_score, 1)))
            raw_res["score"] = clamped_score

            match_schema = MatchScoreSchema(**raw_res)
            return match_schema.model_dump()
        except Exception as e:
            logger.error(f"Error computing match score: {e}. Utilizing fallback scoring engine.")
            fallback_res = self.llm_client._score_heuristically(prompt)
            match_schema = MatchScoreSchema(**fallback_res)
            return match_schema.model_dump()
