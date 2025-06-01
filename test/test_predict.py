import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import predict module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestPredict(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with sample landmark data"""
        # Sample landmark data (21 landmarks for hand)
        self.sample_landmarks = [
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
        
        self.sample_data_with_key = {'landmarks': self.sample_landmarks}
    
    @patch('predict.svm_model')
    def test_predict_with_landmarks_key(self, mock_model):
        """Test predict function with data containing 'landmarks' key"""
        # Mock the SVM model prediction
        mock_model.predict.return_value = ['gesture_class']
        
        from predict import predict
        
        result = predict(self.sample_data_with_key)
        
        # Verify that predict was called and returned expected result
        mock_model.predict.assert_called_once()
        self.assertEqual(result, 'gesture_class')
    
    @patch('predict.svm_model')
    def test_predict_with_direct_landmarks(self, mock_model):
        """Test predict function with direct landmarks list"""
        mock_model.predict.return_value = ['another_gesture']
        
        from predict import predict
        
        result = predict(self.sample_landmarks)
        
        mock_model.predict.assert_called_once()
        self.assertEqual(result, 'another_gesture')
    
    @patch('predict.svm_model')
    def test_normalization_and_scaling(self, mock_model):
        """Test that landmarks are properly normalized and scaled"""
        mock_model.predict.return_value = ['test_gesture']
        
        from predict import predict
        
        # Create a spy to capture the features passed to the model
        def capture_features(features):
            self.captured_features = features
            return ['test_gesture']
        
        mock_model.predict.side_effect = capture_features
        
        predict(self.sample_landmarks)
        
        # Verify features shape (21 landmarks * 3 coordinates = 63 features)
        self.assertEqual(self.captured_features.shape, (1, 63))
        
        # Verify that features are normalized (wrist should be at origin for x,y)
        features_2d = self.captured_features.reshape(21, 3)
        wrist_x, wrist_y = features_2d[0, 0], features_2d[0, 1]
        self.assertAlmostEqual(wrist_x, 0.0, places=10)
        self.assertAlmostEqual(wrist_y, 0.0, places=10)
    
    def test_invalid_landmarks_length(self):
        """Test handling of invalid landmarks data"""
        from predict import predict
        
        # Test with insufficient landmarks
        invalid_landmarks = [{'x': 0.1, 'y': 0.2, 'z': 0.3}]  # Only 1 landmark instead of 21
        
        with self.assertRaises(IndexError):
            predict(invalid_landmarks)
    
    def test_missing_coordinate_keys(self):
        """Test handling of landmarks missing required keys"""
        from predict import predict
        
        # Test with missing 'z' key
        invalid_landmarks = [{'x': 0.1, 'y': 0.2}] * 21
        
        with self.assertRaises(KeyError):
            predict(invalid_landmarks)
    
    @patch('predict.svm_model')
    def test_zero_scale_factor_handling(self, mock_model):
        """Test handling when scale factor becomes zero"""
        mock_model.predict.return_value = ['test_gesture']
        
        from predict import predict
        
        # Create landmarks where mid-finger (index 12) is at the same position as wrist
        zero_scale_landmarks = []
        for i in range(21):
            if i == 0 or i == 12:  # wrist and mid-finger at same position
                zero_scale_landmarks.append({'x': 0.5, 'y': 0.5, 'z': 0.0})
            else:
                zero_scale_landmarks.append({'x': 0.6, 'y': 0.6, 'z': 0.1})
        
        # This should raise a warning about division by zero or handle it gracefully
        with self.assertWarns(RuntimeWarning):
            predict(zero_scale_landmarks)

class TestPredictIntegration(unittest.TestCase):
    """Integration tests that require the actual model file"""
    
    def setUp(self):
        """Check if model file exists"""
        self.model_path = "model/model_svm.pkl"
        self.model_exists = os.path.exists(self.model_path)
    
    @unittest.skipUnless(os.path.exists("model/model_svm.pkl"), "Model file not found")
    def test_predict_with_real_model(self):
        """Test predict function with actual model (if available)"""
        from predict import predict
        
        sample_landmarks = [
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
        
        result = predict(sample_landmarks)
        
        # Verify that result is a string (gesture class)
        self.assertIsInstance(result, (str, int, np.integer))

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)