import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import pyttsx3
import time

# Load trained model
model = joblib.load("sibi_knn.pkl")

# Block warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

# Text-to-Speech
def tts(text, lang):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    voices = engine.getProperty('voices')
    for voice in voices:
        # Create a searchable string from the voice's hidden attributes
        search_str = f"{voice.name} {voice.id} {voice.languages}".lower()
        
        if lang == "ID" and ("indonesia" in search_str or "id-" in search_str or "id_" in search_str):
            engine.setProperty('voice', voice.id)
            break
        elif lang == "EN" and ("english" in search_str or "en-" in search_str or "en_" in search_str):
            engine.setProperty('voice', voice.id)
            break

    engine.say(text)
    engine.runAndWait()

APPEND_DELAY = 2
TTS_DELAY = 5
word_list = []
current_stable_pred = None
stable_start_time = time.time()
last_input_time = time.time()
has_appended = False
current_lang = 'ID'

# Mediapipe configs
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Logo
logo = cv2.imread("aset-03-cropped.png", cv2.IMREAD_UNCHANGED)
if logo is not None:
    # Get original dimensions (height, width)
    orig_h, orig_w = logo.shape[:2]

    target_h = 70
    target_w = int(orig_w * (target_h / orig_h))
    
    # Resize the logo with the proportional dimensions
    logo = cv2.resize(logo, (target_w, target_h))
else:
    print("Not found.")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960) # Height

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

COLOR_ACCENT = (80, 190, 165)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (20, 45, 25)
COLOR_BOX = (80, 190, 165)

while True:
    ret, img = cap.read()
    if not ret:
        print("WARNING: Frame not received... retrying")
        continue

    img = cv2.flip(img, 1)
    h, w, c = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    pred_label = "Waiting..."

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in hand.landmark[:21]])

        if pts.shape == (21, 2):
            pad = 20
            x_min, y_min = np.min(pts, axis=0).astype(int)
            x_max, y_max = np.max(pts, axis=0).astype(int)
            
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLOR_BOX, 1)
            
            
            line_len = 15
            thick = 3
            
            cv2.line(img, (x_min, y_min), (x_min + line_len, y_min), COLOR_BOX, thick)
            cv2.line(img, (x_min, y_min), (x_min, y_min + line_len), COLOR_BOX, thick)
            
            cv2.line(img, (x_max, y_min), (x_max - line_len, y_min), COLOR_BOX, thick)
            cv2.line(img, (x_max, y_min), (x_max, y_min + line_len), COLOR_BOX, thick)
            
            cv2.line(img, (x_min, y_max), (x_min + line_len, y_max), COLOR_BOX, thick)
            cv2.line(img, (x_min, y_max), (x_min, y_max - line_len), COLOR_BOX, thick)
            
            cv2.line(img, (x_max, y_max), (x_max - line_len, y_max), COLOR_BOX, thick)
            cv2.line(img, (x_max, y_max), (x_max, y_max - line_len), COLOR_BOX, thick)

            # center
            center = np.mean(pts, axis=0)
            norm_pts = pts - center

            # scale
            max_dist = np.max(np.linalg.norm(norm_pts, axis=1))
            if max_dist > 0:
                norm_pts /= max_dist

            X = norm_pts.reshape(1, -1)
            pred_label = str(model.predict(X)[0])

        mp_draw.draw_landmarks(
            img, hand, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

    # Text-to-speech
    if pred_label != "Waiting...":
        # If the sign is the same as the last frame
        if pred_label == current_stable_pred:
            # Use the new APPEND_DELAY variable
            if not has_appended and (time.time() - stable_start_time >= APPEND_DELAY):
                word_list.append(pred_label)
                has_appended = True
                last_input_time = time.time() # Reset TTS countdown
        else:
            # The sign changed, reset the timer
            current_stable_pred = pred_label
            stable_start_time = time.time()
            has_appended = False
    else:
        # No hand detected, reset tracking variables
        current_stable_pred = None
        has_appended = False

    if len(word_list) > 0 and (time.time() - last_input_time >= TTS_DELAY):
        final_word = "".join(word_list)
        # Update the args to include current_lang!
        threading.Thread(target=tts, args=(final_word, current_lang), daemon=True).start()
        word_list = []

    banner_height = 80
    cv2.rectangle(img, (0, h - banner_height), (w, h), COLOR_BG, -1)
    cv2.rectangle(img, (0, h - banner_height), (w, h - banner_height + 5), COLOR_ACCENT, -1)
    
    if logo is not None:
        # Get the dimensions of the resized logo
        logo_h, logo_w = logo.shape[:2]
        
        # Calculate position (Middle Top with 10 pixels padding from the top edge)
        pad = 10
        y1, y2 = pad, pad + logo_h
        
        # Find the center of the screen, then offset by half the logo's width to center it
        x1 = (w // 2) - (logo_w // 2)
        x2 = x1 + logo_w

        # If the logo has a transparent (alpha) channel (4 channels: BGRA)
        if logo.shape[2] == 4:
            # Extract the alpha channel and create a mask
            alpha_logo = logo[:, :, 3] / 255.0
            alpha_img = 1.0 - alpha_logo

            # Blend the logo and the background image based on the alpha mask
            for c in range(0, 3):
                img[y1:y2, x1:x2, c] = (alpha_logo * logo[:, :, c] +
                                        alpha_img * img[y1:y2, x1:x2, c])
        else:
            # If the logo is a flat image (like a JPG), just paste it directly
            img[y1:y2, x1:x2] = logo

    font_small = cv2.FONT_HERSHEY_SIMPLEX
    lang_text = f"Lang: {current_lang} (Press 'L')"
    # Put it in the top left corner
    cv2.putText(img, lang_text, (20, 40), font_small, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, lang_text, (20, 40), font_small, 0.8, COLOR_ACCENT, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_DUPLEX
    display_text = "".join(word_list)

    if len(display_text) > 0:
        cv2.putText(img, f"Input: {display_text}", (30, h - 110), font, 1.5, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, f"Input: {display_text}", (30, h - 110), font, 1.5, COLOR_ACCENT, 2, cv2.LINE_AA)


    text = f"SIBI: {pred_label}"

    if has_appended:
        text += " (ADDED!)"
        text_color = (0, 255, 0) # Green if locked in
    else:
        text_color = COLOR_TEXT  # White otherwise

    # Draw the bottom banner text
    cv2.putText(img, text, (30, h - 25), font, 1.2, (0, 0, 0), 4, cv2.LINE_AA) # Shadow
    cv2.putText(img, text, (30, h - 25), font, 1.2, text_color, 2, cv2.LINE_AA) # Main text

    cv2.imshow("SIBI Gesture Classifier", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 8 or key == 127: # ASCII codes for Backspace (Windows/Mac)
        if len(word_list) > 0:
            word_list.pop()
            last_input_time = time.time()
    elif key == ord('l'): # <--- ADD THIS BLOCK
        # Toggle between ID and EN
        if current_lang == "ID":
            current_lang = "EN"
        else:
            current_lang = "ID"

cap.release()
cv2.destroyAllWindows()