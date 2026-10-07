import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String(255), nullable=False)
    description_text = Column(Text, nullable=False)
    requirements = Column(JSON, nullable=True, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

    match_results = relationship("MatchResult", back_populates="job_description", cascade="all, delete-orphan")


class CandidateResume(Base):
    __tablename__ = "candidate_resumes"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    candidate_name = Column(String(255), nullable=True)
    contact_info = Column(JSON, nullable=True, default=dict)
    raw_text = Column(Text, nullable=False)
    skills = Column(JSON, nullable=True, default=list)
    experience = Column(JSON, nullable=True, default=list)
    education = Column(JSON, nullable=True, default=list)
    extraction_status = Column(String(50), default="COMPLETED")
    created_at = Column(DateTime, default=datetime.utcnow)

    match_results = relationship("MatchResult", back_populates="candidate", cascade="all, delete-orphan")


class MatchResult(Base):
    __tablename__ = "match_results"

    id = Column(String, primary_key=True, default=generate_uuid)
    candidate_id = Column(String, ForeignKey("candidate_resumes.id", ondelete="CASCADE"), nullable=False)
    job_description_id = Column(String, ForeignKey("job_descriptions.id", ondelete="CASCADE"), nullable=False)
    score = Column(Float, nullable=False)
    justification = Column(Text, nullable=False)
    pros = Column(JSON, nullable=True, default=list)
    cons = Column(JSON, nullable=True, default=list)
    matching_skills = Column(JSON, nullable=True, default=list)
    missing_skills = Column(JSON, nullable=True, default=list)
    timestamp = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("CandidateResume", back_populates="match_results")
    job_description = relationship("JobDescription", back_populates="match_results")
