import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import logging
import fitz  # PyMuPDF
import docx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
load_dotenv()

# Simplified Pydantic models
class GradingInput(BaseModel):
    question: str
    text: str
    rubric: str

class GradingResult(BaseModel):
    student_id: str
    assignment_id: str
    grades: Dict[str, str]
    feedback: str

# LangChain setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

grading_prompt = ChatPromptTemplate.from_template("""
Grade the following assignment based on the provided rubric and the question. Provide detailed feedback with marks for each section:

Question:
{question}

Rubric Criteria:
{rubric}

Assignment Text:
{text}

Provide the marks and feedback for each criterion. Use the format 'Criterion: Score' for each graded item, followed by detailed feedback.
""")

chain = LLMChain(llm=llm, prompt=grading_prompt)

# File reading functions
async def read_pdf(file: UploadFile) -> str:
    content = await file.read()
    doc = fitz.open(stream=content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

async def read_docx(file: UploadFile) -> str:
    content = await file.read()
    doc = docx.Document(content)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)

async def read_rubric_file(file: UploadFile) -> str:
    content_type = file.content_type.lower()
    if 'pdf' in content_type:
        return await read_pdf(file)
    elif 'word' in content_type or 'docx' in content_type:
        return await read_docx(file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

# FastAPI endpoint
@app.post("/gradeAssignment", response_model=GradingResult)
async def grade_assignment_endpoint(
    background_tasks: BackgroundTasks,
    student_id: str = Form(...),
    assignment_id: str = Form(...),
    question: str = Form(...),
    text: str = Form(...),
    rubric: UploadFile = File(...)
):
    try:
        rubric_text = await read_rubric_file(rubric)
        grading_input = GradingInput(
            question=question,
            text=text,
            rubric=rubric_text
        )
        
        result = await chain.arun(grading_input.dict())

        # Parse the result string into grades and feedback
        lines = result.split('\n')
        grades = {}
        feedback = []
        for line in lines:
            if ':' in line:
                criterion, score = line.split(':', 1)
                grades[criterion.strip()] = score.strip()
            else:
                feedback.append(line)

        # Log the grading result (you might want to store this in a database)
        background_tasks.add_task(logger.info, f"Grading completed for student {student_id}, assignment {assignment_id}")

        return GradingResult(
            student_id=student_id,
            assignment_id=assignment_id,
            grades=grades,
            feedback='\n'.join(feedback)
        )
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Grading error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during grading")

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)