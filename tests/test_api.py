from fastapi.testclient import TestClient
from src.api import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Food Image Classification API!"}

if __name__ == "__main__":
    test_read_root()