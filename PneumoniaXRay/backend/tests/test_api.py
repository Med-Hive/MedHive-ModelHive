import pytest
from fastapi.testclient import TestClient
from app.main import app
import time
import os
from pathlib import Path

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["timestamp"], float)


def test_predict_valid_image():
    """Test prediction with a valid X-ray image"""
    # Use a test image from the dataset
    test_image_path = os.path.join(
        "data", "chest_xray", "test", "PNEUMONIA", "person1_bacteria_1.jpeg"
    )
    
    # Skip if the test file doesn't exist
    if not os.path.exists(test_image_path):
        pytest.skip(f"Test image not found at {test_image_path}")
    
    with open(test_image_path, "rb") as image_file:
        files = {"file": ("test_xray.jpeg", image_file, "image/jpeg")}
        response = client.post("/api/v1/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "diagnosis" in data
    assert "probability" in data
    assert "timestamp" in data
    assert data["diagnosis"] in ["Normal", "Pneumonia"]
    assert 0 <= data["probability"] <= 1
    assert isinstance(data["timestamp"], float)

def test_predict_invalid_file_type():
    """Test prediction with an invalid file type (text instead of image)"""
    # Create a text file for testing
    with open("test_file.txt", "w") as f:
        f.write("This is not an image file")
    
    with open("test_file.txt", "rb") as text_file:
        files = {"file": ("test_file.txt", text_file, "text/plain")}
        response = client.post("/api/v1/predict", files=files)
    
    # Clean up
    os.remove("test_file.txt")
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"] == "INVALID_FILE_TYPE"

def test_predict_without_file():
    """Test prediction without providing a file"""
    response = client.post("/api/v1/predict")
    assert response.status_code == 422  # Validation error