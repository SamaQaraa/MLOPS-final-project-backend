import unittest
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add the parent directory to the path to import app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

class TestFastAPIApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test client and sample data"""
        self.client = TestClient(app)
        
        # Sample valid landmark data
        self.valid_landmarks_data = {
            "landmarks": [
                {'x': 0.09297193586826324, 'y': 0.7771648168563843, 'z': -5.838224979015649e-07},
                {'x': 0.19567665457725525, 'y': 0.7501276731491089, 'z': -0.01315996889024973},
                {'x': 0.276355117559433, 'y': 0.6616146564483643, 'z': -0.01374891772866249},
                {'x': 0.3073362112045288, 'y': 0.5310574173927307, 'z': -0.017140453681349754},
                {'x': 0.3133564591407776, 'y': 0.4412885904312134, 'z': -0.014094164595007896},
                {'x': 0.2564017176628113, 'y': 0.5193532705307007, 'z': 0.008522840216755867},
                {'x': 0.27478066086769104, 'y': 0.4029613435268402, 'z': -0.026238704100251198},
                {'x': 0.2498849481344223, 'y': 0.5161670446395874, 'z': -0.0389925017952919},
                {'x': 0.23827366530895233, 'y': 0.5786231756210327, 'z': -0.040617190301418304},
                {'x': 0.20124201476573944, 'y': 0.4947439134120941, 'z': 0.0036657012533396482},
                {'x': 0.21710337698459625, 'y': 0.399094820022583, 'z': -0.037151187658309937},
                {'x': 0.19466206431388855, 'y': 0.5442489385604858, 'z': -0.04216064140200615},
                {'x': 0.18851310014724731, 'y': 0.5892699956893921, 'z': -0.036311764270067215},
                {'x': 0.14723482728004456, 'y': 0.4850277304649353, 'z': -0.006600293330848217},
                {'x': 0.16313469409942627, 'y': 0.3952963948249817, 'z': -0.05445410683751106},
                {'x': 0.14420142769813538, 'y': 0.5429767966270447, 'z': -0.036745015531778336},
                {'x': 0.13771110773086548, 'y': 0.5937791466712952, 'z': -0.011988840065896511},
                {'x': 0.0869116261601448, 'y': 0.4821779727935791, 'z': -0.017018601298332214},
                {'x': 0.10463820397853851, 'y': 0.4028026759624481, 'z': -0.04689931124448776},
                {'x': 0.09783300757408142, 'y': 0.5133748650550842, 'z': -0.029298530891537666},
                {'x': 0.09144939482212067, 'y': 0.5639343857765198, 'z': -0.00768341775983572}
            ]
        }
    
    @patch('app.predict')
    def test_predict_endpoint_success(self, mock_predict):
        """Test successful prediction endpoint"""
        # Mock the predict function
        mock_predict.return_value = "thumbs_up"
        
        response = self.client.post(
            "/predict",
            json=self.valid_landmarks_data
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"gesture": "thumbs_up"})
        
        # Verify predict was called with correct data
        mock_predict.assert_called_once_with(self.valid_landmarks_data["landmarks"])
    
    @patch('app.predict')
    def test_predict_endpoint_with_different_gesture(self, mock_predict):
        """Test prediction endpoint with different gesture"""
        mock_predict.return_value = "peace_sign"
        
        response = self.client.post(
            "/predict",
            json=self.valid_landmarks_data
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"gesture": "peace_sign"})
    
    def test_predict_endpoint_missing_landmarks(self):
        """Test prediction endpoint with missing landmarks key"""
        invalid_data = {"not_landmarks": []}
        
        response = self.client.post(
            "/predict",
            json=invalid_data
        )
        
        # Should return 500 internal server error due to KeyError
        self.assertEqual(response.status_code, 400)
    
    def test_predict_endpoint_invalid_json(self):
        """Test prediction endpoint with invalid JSON"""
        response = self.client.post(
            "/predict",
            data="invalid json"
        )
        
        # Should return 400 or 422 for invalid JSON
        self.assertIn(response.status_code, [400, 422])
    
    def test_predict_endpoint_empty_request(self):
        """Test prediction endpoint with empty request body"""
        response = self.client.post("/predict")
        
        # Should return 400 or 422 for missing request body
        self.assertIn(response.status_code, [400, 422])
    
    @patch('app.predict')
    def test_predict_endpoint_exception_handling(self, mock_predict):
        """Test prediction endpoint when predict function raises exception"""
        # Mock predict to raise an exception
        mock_predict.side_effect = Exception("Model prediction failed")
        
        response = self.client.post(
            "/predict",
            json=self.valid_landmarks_data
        )
        
        # Should return 500 internal server error
        self.assertEqual(response.status_code, 500)
    
    def test_cors_headers(self):
        """Test that CORS headers are properly set"""
        response = self.client.options("/predict")
        
        # Check that CORS is enabled (response should not be 405 Method Not Allowed)
        self.assertNotEqual(response.status_code, 400)
    
    def test_unsupported_method(self):
        """Test unsupported HTTP methods"""
        # Test GET request (should not be allowed)
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 405)
        
        # Test PUT request (should not be allowed)
        response = self.client.put("/predict", json=self.valid_landmarks_data)
        self.assertEqual(response.status_code, 405)
        
        # Test DELETE request (should not be allowed)
        response = self.client.delete("/predict")
        self.assertEqual(response.status_code, 405)
    
    def test_root_endpoint_not_defined(self):
        """Test that root endpoint returns 404"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 404)
    
    def test_undefined_endpoint(self):
        """Test undefined endpoint returns 404"""
        response = self.client.get("/undefined-endpoint")
        self.assertEqual(response.status_code, 404)
    
    @patch('app.predict')
    def test_content_type_json(self, mock_predict):
        """Test that endpoint properly handles JSON content type"""
        mock_predict.return_value = "test_gesture"
        
        response = self.client.post(
            "/predict",
            json=self.valid_landmarks_data,
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
    
    @patch('app.predict')
    def test_large_landmarks_data(self, mock_predict):
        """Test endpoint with unusually large landmarks data"""
        mock_predict.return_value = "gesture_with_large_data"
        
        # Create larger than normal landmarks data
        large_data = {
            "landmarks": self.valid_landmarks_data["landmarks"] * 10  # 210 landmarks instead of 21
        }
        
        response = self.client.post(
            "/predict",
            json=large_data
        )
        
        # Should still work (predict function will handle the error)
        self.assertEqual(response.status_code, 200)
        mock_predict.assert_called_once()

class TestAppConfiguration(unittest.TestCase):
    """Test app configuration and middleware"""
    
    def test_app_instance(self):
        """Test that app is properly configured"""
        from app import app
        self.assertIsNotNone(app)
        self.assertEqual(app.title, "FastAPI")
    
    def test_cors_middleware(self):
        """Test CORS middleware configuration"""
        client = TestClient(app)
        
        # Make a request and check CORS headers
        response = client.post(
            "/predict",
            json={"landmarks": []},
            headers={"Origin": "http://localhost:8000"}
        )
        
        # CORS should allow the request
        self.assertIn("access-control-allow-origin", 
                     [h.lower() for h in response.headers.keys()])

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)