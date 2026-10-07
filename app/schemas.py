from typing import List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field

class ExperienceItem(BaseModel):
    role: str = ""
    company: str = ""
    duration: str = ""
    responsibilities: str = ""

class EducationItem(BaseModel):
    degree: str = ""
    institution: str = ""
    year: str = ""

class CandidateExtractionSchema(BaseModel):
    candidate_name: Optional[str] = "Unknown Candidate"
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)

class JobDescriptionCreate(BaseModel):
    title: str
    description_text: str

class JobDescriptionResponse(BaseModel):
    id: str
    title: str
    description_text: str
    created_at: datetime

    class Config:
        from_attributes = True

class CandidateResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    candidate_name: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[Dict[str, Any]] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime

    class Config:
        from_attributes = True

class MatchScoreSchema(BaseModel):
    score: float = Field(..., ge=1.0, le=10.0, description="Match score from 1 to 10")
    justification: str = Field(..., description="Human readable justification explaining fit")
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    matching_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)

class ShortlistCandidateResponse(BaseModel):
    candidate_id: str
    filename: str
    candidate_name: Optional[str] = "Unknown Candidate"
    score: float
    justification: str
    pros: List[str]
    cons: List[str]
    matching_skills: List[str]
    missing_skills: List[str]
    skills: List[str]
    experience: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    timestamp: datetime

    class Config:
        from_attributes = True

class BatchScreenRequest(BaseModel):
    job_id: str
    candidate_ids: Optional[List[str]] = None

class BatchScreenResponse(BaseModel):
    job_id: str
    job_title: str
    total_screened: int
    shortlist: List[ShortlistCandidateResponse]
