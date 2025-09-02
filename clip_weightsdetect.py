import cv2
import mediapipe as mp
import torch
import clip
from PIL import Image
import time
import numpy as np
from collections import Counter

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Categories
categories = ["dumbbell", "hand", "bottle"]
text_tokens = clip.tokenize(categories).to(device)

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("dumbell1.mp4")

def classify_frame(frame, bbox):
    """Crop around bbox, run CLIP classification, return probabilities"""
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:  # fallback if crop fails
        crop = frame

    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    image_preprocessed = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image_preprocessed, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    return probs

print("Starting... Show both wrists in the camera!")

wrist_visible_start = None
prediction_window_active = False
predictions = []
last_prediction_time = None
prediction_window_start = None

CONF_THRESHOLD = 0.60  # Confidence cutoff

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #resizing video screen 
    max_height = 720
    h, w = frame.shape[:2]
    if h > max_height:
        scale = max_height / float(h)
        frame = cv2.resize(frame, (int(w * scale), max_height))
    #end resize 
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    both_wrists_visible = False
    wrist_bboxes = []

    if results.multi_hand_landmarks:
        wrist_count = 0
        h, w, _ = frame.shape

        for hand_landmarks in results.multi_hand_landmarks:
            # Get all landmark coordinates
            x_coords = [int(w * lm.x) for lm in hand_landmarks.landmark]
            y_coords = [int(h * lm.y) for lm in hand_landmarks.landmark]

            # Adaptive bounding box with margin
            margin = 40
            x1, y1 = max(0, min(x_coords) - margin), max(0, min(y_coords) - margin)
            x2, y2 = min(w, max(x_coords) + margin), min(h, max(y_coords) + margin)

            wrist_bboxes.append((x1, y1, x2, y2))

            # Draw bbox for visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if wrist landmark is valid
            wrist = hand_landmarks.landmark[0]
            if 0 <= wrist.x <= 1 and 0 <= wrist.y <= 1:
                wrist_count += 1

        both_wrists_visible = wrist_count >= 2

    if both_wrists_visible:
        if wrist_visible_start is None:
            wrist_visible_start = time.time()
        elif not prediction_window_active and time.time() - wrist_visible_start >= 5:
            print("Wrist stable for 5 sec, starting prediction window...")
            prediction_window_active = True
            predictions = []
            last_prediction_time = time.time()
            prediction_window_start = time.time()
    else:
        wrist_visible_start = None

    if prediction_window_active:
        if len(predictions) < 5 and time.time() - last_prediction_time >= 2:
            # Use first hand's adaptive bbox
            if wrist_bboxes:
                probs = classify_frame(frame, wrist_bboxes[0])
                pred_idx = int(np.argmax(probs))
                confidence = probs[pred_idx]

                if confidence >= CONF_THRESHOLD:
                    pred_label = categories[pred_idx]
                else:
                    pred_label = "uncertain"

                predictions.append(pred_label)
                print(f"Sample Prediction {len(predictions)}: {pred_label} (probs: {probs})")

            last_prediction_time = time.time()

        if len(predictions) == 5:
            # Majority vote
            final_prediction = Counter(predictions).most_common(1)[0][0]
            print("\n========== FINAL PREDICTION ==========")
            print(f"Predicted: {final_prediction}")
            print("=====================================\n")
            break

    cv2.imshow("Wrist Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
