from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
import docx
import json
from groq_grading_chain import GroqGradingChain  # Updated import for Groq AI

app = FastAPI()

# Initialize GroqGradingChain with your API key
groq_grading_chain = GroqGradingChain(api_key="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")

@app.post("/gradeAssignment")
async def grade_assignment_endpoint(
    student_id: str = Form(...),
    assignment_id: str = Form(...),
    question: str = Form(...),
    text: str = Form(...),
    rubric: UploadFile = File(...)
):
    try:
        assignment_data = {
            "student_id": student_id,
            "assignment_id": assignment_id,
            "question": question,
            "text": text
        }
        rubric_text = await read_rubric_file(rubric)
        result = groq_grading_chain({
            'question': assignment_data['question'],
            'text': assignment_data['text'],
            'rubric': rubric_text
        })

        return {
            "student_id": assignment_data['student_id'],
            "assignment_id": assignment_data['assignment_id'],
            "grades": result['grades'],
            "feedback": result['feedback']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def read_rubric_file(file: UploadFile):
    if file.content_type == 'application/pdf':
        return read_pdf(file)
    elif file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return read_docx(file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

def read_pdf(file):
    doc = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def read_docx(file):
    doc = docx.Document(file.file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

