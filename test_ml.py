import pytest
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app (adjust import as needed)

client = TestClient(app)

def test_root_endpoint():
    """
    Test that the root endpoint returns the welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Welcome to the ML inference API!"

def test_model_inference():
    """
    Test model inference endpoint returns a result for valid input.
    """
    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/data/", json=data)
    assert response.status_code == 200
    assert "result" in response.json()

def test_invalid_data():
    """
    Test inference endpoint with invalid data returns error.
    """
    data = {
        # Missing required fields, or bad types
        "age": "not_an_int",
    }
    response = client.post("/data/", json=data)
    assert response.status_code == 422  # Unprocessable Entity