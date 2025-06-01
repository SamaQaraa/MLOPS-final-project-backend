import numpy as np
import joblib  # For loading the trained SVM model
import pickle

# Load the trained SVM model
with open("model/model_svm.pkl", "rb") as f:
    svm_model = pickle.load(f)

def predict(landmarks_data):
    # landmarks_data is now a list of dictionaries, not MediaPipe objects
    landmarks_list = landmarks_data['landmarks'] if isinstance(landmarks_data, dict) else landmarks_data
    
    # Extract (x, y, z) coordinates from dictionary format
    landmarks = np.array([(lm['x'], lm['y'], lm['z']) for lm in landmarks_list])

    # Normalize: Recenter based on wrist position (landmark 0)
    wrist_x, wrist_y, wrist_z = landmarks[0]
    landmarks[:, 0] -= wrist_x  
    landmarks[:, 1] -= wrist_y  

    # Scale only x and y using the mid-finger tip (landmark 12)
    mid_finger_x, mid_finger_y, _ = landmarks[12] 
    scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
    landmarks[:, 0] /= scale_factor  
    landmarks[:, 1] /= scale_factor  

    # Flatten the features for SVM
    features = landmarks.flatten().reshape(1, -1)

    # Predict using SVM
    prediction = svm_model.predict(features)[0]

    return prediction