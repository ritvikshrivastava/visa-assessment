from io import BytesIO

import pypdf
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query

from visa_assessment.judge import judge_eligibility

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to You O-1A Visa Assessment Using LLMs"}


@app.post("/assess_visa")
async def assessment(cv: UploadFile = File(...), azure: bool = Query(False)):
    """
    API endpoint to access the eligibility assessment logic in judge.py
    """
    extension = cv.filename.split(".")[-1].lower()
    cv_data = await cv.read()

    if extension == "pdf":
        with BytesIO(cv_data) as pdf_file:
            pdf_reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    elif extension in ("txt", "docx"):
        text = cv_data.decode("utf-8")

    assessment_result = await judge_eligibility(cv_text=text, azure=azure)
    return assessment_result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
