# Assessment of Visa Eligibility
An AI system (using OpenAI's GPT-4o) to parse information from a resume/cv file and assess 
whether the candidate qualifies for an O1A visa or not.

Detailed description in ``docs.md``

## Installation of the module and all requirements
```angular2html
poetry install
```

## Setting Environment Variables for calling GPT-4o
### via OpenAI
```angular2html
export OPENAI_API_KEY=<>
```
### via Azure OpenAI
```angular2html
export AZURE_OPENAI_API_KEY=<>
export AZURE_OPENAI_ENDPOINT=<>
```

You can also explicitly set them in ``visa_assessment/judge.py``

## Starting the FastAPI server
```angular2html
poetry run uvicorn api:app --reload
```

## Making Requests against the server

```angular2html
curl -X POST "http://localhost:8000/assess_visa" \
-H "Content-Type: multipart/form-data" \
-F "cv=@test_cv.txt" | jq .
```

### if using Azure
```angular2html
curl -X POST "http://localhost:8000/assess_visa?azure=True" \
-H "Content-Type: multipart/form-data" \
-F "cv=@test_cv.txt" | jq .
```

     