from locust import HttpUser, task, between

class FastAPILoadTestUser(HttpUser):
    wait_time = between(1, 2)  # Simulate a delay between requests
    host = "http://127.0.0.1:8000"  # Set the host to your FastAPI app's URL

    @task
    def test_predict(self):
        """Send a test image to the /predict/ endpoint."""
        with open("test_image/sample.jpg", "rb") as img:  # Replace with your test image path
            self.client.post("/predict/", files={"file": img})
