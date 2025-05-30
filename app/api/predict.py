import cv2
import mediapipe as mp
import numpy as np
import joblib

def predict(model_path="model_svm.pkl"):
    # Load the model
    svm_model = joblib.load(model_path)

    # Initialize Mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                wrist = landmarks[0]
                landmarks[:, :2] -= wrist[:2]

                scale = np.linalg.norm(landmarks[12][:2])
                if scale != 0:
                    landmarks[:, :2] /= scale

                features = landmarks.flatten().reshape(1, -1)
                prediction = svm_model.predict(features)[0]

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f'Prediction: {prediction}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
