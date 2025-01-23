from fastapi.testclient import TestClient
from src.api import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Food Image Classification API!"}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["model_loaded"] is True

if __name__ == "__main__":
    test_read_root()
    test_health_check()
