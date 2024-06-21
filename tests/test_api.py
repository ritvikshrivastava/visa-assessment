import pytest
import requests
from fastapi.testclient import TestClient
from visa_assessment.api import app

client = TestClient(app)


@pytest.fixture
def filepath():
    return "test_cv.txt"


def test_assess_cv_endpoint_with_file_upload(filepath):
    url = f"http://localhost:8000/assess_visa"
    with open(filepath, "rb") as f:
        files = {"cv": ("sample_cv.txt", f, "text/plain")}
        response = requests.post(url, files=files)
        print(response.__dict__)

    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    assert "rating" in result
    assert isinstance(result["results"], dict)
    assert result["rating"] in ["low", "medium", "high"]