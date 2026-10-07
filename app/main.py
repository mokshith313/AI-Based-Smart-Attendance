import os
import logging
from typing import List, Optional
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.models import JobDescription, CandidateResume, MatchResult
from app import schemas
from app.parser import parse_resume_bytes
from app.extractor import ResumeExtractor
from app.scorer import ResumeScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_screener")

# Initialize DB tables on startup
init_db()

app = FastAPI(
    title="Smart Resume Screener API",
    description="AI-powered resume ingestion, structured parsing, and LLM-based semantic match scoring engine.",
    version="1.0.0"
)

# CORS middleware for development/frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = ResumeExtractor()
scorer = ResumeScorer()

# --- JOB DESCRIPTION ENDPOINTS ---

@app.post("/api/v1/jobs", response_model=schemas.JobDescriptionResponse, status_code=status.HTTP_201_CREATED)
def create_job_description(job_in: schemas.JobDescriptionCreate, db: Session = Depends(get_db)):
    """Creates a new Job Description record."""
    job = JobDescription(
        title=job_in.title,
        description_text=job_in.description_text
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job

@app.get("/api/v1/jobs", response_model=List[schemas.JobDescriptionResponse])
def list_job_descriptions(db: Session = Depends(get_db)):
    """Lists all saved Job Descriptions."""
    return db.query(JobDescription).order_by(JobDescription.created_at.desc()).all()

@app.get("/api/v1/jobs/{job_id}", response_model=schemas.JobDescriptionResponse)
def get_job_description(job_id: str, db: Session = Depends(get_db)):
    """Retrieves a specific Job Description by ID."""
    job = db.query(JobDescription).filter(JobDescription.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found")
    return job

# --- RESUME INGESTION & EXTRACTION ENDPOINTS ---

@app.post("/api/v1/resumes/upload", response_model=List[schemas.CandidateResponse])
async def upload_resumes(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """
    Ingests multiple PDF or TXT resume files, extracts raw text, 
    parses structured profile data (skills, experience, education), and persists candidates.
    """
    candidates = []
    for file in files:
        filename = file.filename or "uploaded_resume.txt"
        contents = await file.read()
        
        # 1. Raw Text Parsing
        raw_text, file_type = parse_resume_bytes(contents, filename)
        if not raw_text.strip():
            logger.warning(f"File {filename} produced empty raw text.")
            raw_text = "No readable text content extracted."

        # 2. Structured Extraction
        extracted_data = extractor.extract(raw_text)
        candidate_name = extracted_data.get("candidate_name") or filename.split('.')[0].replace('_', ' ').replace('-', ' ').title()

        # 3. DB Persistence
        candidate = CandidateResume(
            filename=filename,
            file_type=file_type,
            candidate_name=candidate_name,
            contact_info={
                "email": extracted_data.get("contact_email"),
                "phone": extracted_data.get("contact_phone")
            },
            raw_text=raw_text,
            skills=extracted_data.get("skills", []),
            experience=extracted_data.get("experience", []),
            education=extracted_data.get("education", []),
            extraction_status="COMPLETED"
        )
        db.add(candidate)
        db.commit()
        db.refresh(candidate)
        candidates.append(candidate)

    return candidates

@app.get("/api/v1/candidates", response_model=List[schemas.CandidateResponse])
def list_candidates(db: Session = Depends(get_db)):
    """Lists all uploaded/parsed candidates."""
    return db.query(CandidateResume).order_by(CandidateResume.created_at.desc()).all()

@app.get("/api/v1/candidates/{candidate_id}", response_model=schemas.CandidateResponse)
def get_candidate(candidate_id: str, db: Session = Depends(get_db)):
    """Retrieves candidate profile and extracted structured data."""
    candidate = db.query(CandidateResume).filter(CandidateResume.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate

@app.delete("/api/v1/candidates/{candidate_id}")
def delete_candidate(candidate_id: str, db: Session = Depends(get_db)):
    """Deletes a candidate record and associated match results."""
    candidate = db.query(CandidateResume).filter(CandidateResume.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    db.delete(candidate)
    db.commit()
    return {"message": "Candidate deleted successfully", "id": candidate_id}

# --- LLM MATCH SCORING & SHORTLIST ENDPOINTS ---

@app.post("/api/v1/screen", response_model=schemas.BatchScreenResponse)
def screen_candidates(req: schemas.BatchScreenRequest, db: Session = Depends(get_db)):
    """
    Computes semantic fit scores (1-10) and evidence-based justifications comparing 
    uploaded candidates against the specified Job Description.
    """
    job = db.query(JobDescription).filter(JobDescription.id == req.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found")

    # Determine candidates to screen
    query = db.query(CandidateResume)
    if req.candidate_ids:
        query = query.filter(CandidateResume.id.in_(req.candidate_ids))
    candidates = query.all()

    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates available to screen")

    results = []
    for cand in candidates:
        candidate_data = {
            "candidate_name": cand.candidate_name,
            "skills": cand.skills or [],
            "experience": cand.experience or [],
            "education": cand.education or []
        }

        # LLM Semantic Match Evaluation
        evaluation = scorer.score_candidate(candidate_data, job.description_text)

        # Check if match result already exists, update or create new
        existing_result = db.query(MatchResult).filter(
            MatchResult.candidate_id == cand.id,
            MatchResult.job_description_id == job.id
        ).first()

        if existing_result:
            existing_result.score = evaluation["score"]
            existing_result.justification = evaluation["justification"]
            existing_result.pros = evaluation["pros"]
            existing_result.cons = evaluation["cons"]
            existing_result.matching_skills = evaluation["matching_skills"]
            existing_result.missing_skills = evaluation["missing_skills"]
            match_rec = existing_result
        else:
            match_rec = MatchResult(
                candidate_id=cand.id,
                job_description_id=job.id,
                score=evaluation["score"],
                justification=evaluation["justification"],
                pros=evaluation["pros"],
                cons=evaluation["cons"],
                matching_skills=evaluation["matching_skills"],
                missing_skills=evaluation["missing_skills"]
            )
            db.add(match_rec)
        
        db.commit()
        db.refresh(match_rec)

    # Return ranked shortlist sorted by score descending
    return get_job_shortlist(job_id=job.id, db=db)


@app.get("/api/v1/shortlist/{job_id}", response_model=schemas.BatchScreenResponse)
def get_job_shortlist(job_id: str, db: Session = Depends(get_db)):
    """
    Returns shortlist of candidates ranked by match score (descending) for a job description.
    """
    job = db.query(JobDescription).filter(JobDescription.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found")

    matches = db.query(MatchResult).filter(
        MatchResult.job_description_id == job_id
    ).order_by(MatchResult.score.desc()).all()

    shortlist_items = []
    for m in matches:
        cand = db.query(CandidateResume).filter(CandidateResume.id == m.candidate_id).first()
        if not cand:
            continue
        shortlist_items.append(schemas.ShortlistCandidateResponse(
            candidate_id=cand.id,
            filename=cand.filename,
            candidate_name=cand.candidate_name or cand.filename,
            score=m.score,
            justification=m.justification,
            pros=m.pros or [],
            cons=m.cons or [],
            matching_skills=m.matching_skills or [],
            missing_skills=m.missing_skills or [],
            skills=cand.skills or [],
            experience=cand.experience or [],
            education=cand.education or [],
            timestamp=m.timestamp
        ))

    return schemas.BatchScreenResponse(
        job_id=job.id,
        job_title=job.title,
        total_screened=len(shortlist_items),
        shortlist=shortlist_items
    )

# --- STATIC DASHBOARD SERVING ---

# Serve static directory if present
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serves the main Recruiter Dashboard single-page application."""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return JSONResponse({
        "name": "Smart Resume Screener API",
        "status": "Online",
        "docs_url": "/docs"
    })
