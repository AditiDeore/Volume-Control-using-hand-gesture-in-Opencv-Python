import cv2
import mediapipe as mp

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the index of your camera if different

while cap.isOpened():
    # Read frames from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # If hands are detected, draw landmarks and connections
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of thumb, index finger, and middle finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate distances between thumb and index finger, and thumb and middle finger
            distance_thumb_index = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            distance_thumb_middle = ((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)**0.5

            # Define thresholds for pinch gestures
            pinch_threshold = 0.05

            # Check for increasing volume gesture (thumb and index finger pinched together, middle finger raised)
            if distance_thumb_index < pinch_threshold and middle_tip.y < index_tip.y:
                cv2.putText(frame, "Increasing Volume", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Perform action to increase volume

            # Check for decreasing volume gesture (thumb and middle finger pinched together, index finger raised)
            elif distance_thumb_middle < pinch_threshold and index_tip.y < middle_tip.y:
                cv2.putText(frame, "Decreasing Volume", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Perform action to decrease volume

    # Display the frame with detected landmarks
    cv2.imshow('Hand Gestures', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
