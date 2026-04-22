# Import libraries
import os
import cv2
import mediapipe as mp
import time
import joblib
import edge_tts
import asyncio
import pygame
import tempfile
import threading
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("sibi_knn.pkl")

# Block warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

# Text-to-Speech
APPEND_DELAY = 1.5
TTS_DELAY = 3
sentence_list = []   # List of completed words (strings)
word_list = []       # Current word being built (list of letters)
current_stable_pred = None
stable_start_time = time.time()
last_input_time = time.time()
has_appended = False

# Language & TTS Voice Config
# Press 'L' to toggle between Indonesian and English
LANG_CONFIG = {
    "ID": {"voice": "id-ID-ArdiNeural", "label": "ID - Indonesia"},
    "EN": {"voice": "en-US-GuyNeural",  "label": "EN - English"},
}
current_lang = "ID"  # Default language

# Initialize pygame mixer for audio playback
pygame.mixer.init()

def tts(text, voice):
    """Speak text using edge-tts with the given voice."""
    async def _speak():
        # Create a temp file for the audio
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_path = tmp.name
        tmp.close()
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(tmp_path)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        finally:
            pygame.mixer.music.unload()
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    asyncio.run(_speak())

# Mediapipe configs
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

COLOR_ACCENT = (80, 190, 165)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (20, 45, 25)
COLOR_BOX = (80, 190, 165)

# Logo
# logo = cv2.imread("aset-03-cropped.png", cv2.IMREAD_UNCHANGED)
# if logo is not None:
#     # Get original dimensions (height, width)
#     orig_h, orig_w = logo.shape[:2]

#     target_h = 70
#     target_w = int(orig_w * (target_h / orig_h))
    
#     # Resize the logo with the proportional dimensions
#     logo = cv2.resize(logo, (target_w, target_h))
# else:
#     print("Logo found.")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960) # Height

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

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

        # Hand
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
            X_df = pd.DataFrame(X, columns=model.feature_names_in_)
            pred_label = str(model.predict(X_df)[0])

        mp_draw.draw_landmarks(
            img, hand, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

    # Text-to-speech
    if pred_label != "Waiting...":
        # Hand is detected! Keep resetting the TTS timer so it doesn't speak.
        last_input_time = time.time() 

        if pred_label == current_stable_pred:
            if not has_appended and (time.time() - stable_start_time >= APPEND_DELAY):
                word_list.append(pred_label)
                has_appended = True
        else:
            current_stable_pred = pred_label
            stable_start_time = time.time()
            has_appended = False
    else:
        # No hand detected! (Timer naturally starts counting up from last_input_time)
        current_stable_pred = None
        has_appended = False

    # Check for TTS Trigger:
    # 1. List is not empty
    # 2. Hand is NOT on screen
    # 3. It has been TTS_DELAY seconds since the hand dropped
    # Build full sentence for TTS and display
    if len(word_list) > 0:
        current_word = "".join(word_list)
        full_sentence = " ".join(sentence_list + [current_word])
    elif len(sentence_list) > 0:
        full_sentence = " ".join(sentence_list)
    else:
        full_sentence = ""

    if len(full_sentence) > 0 and pred_label == "Waiting..." and (time.time() - last_input_time >= TTS_DELAY):
        voice = LANG_CONFIG[current_lang]["voice"]
        threading.Thread(target=tts, args=(full_sentence, voice), daemon=True).start()
        sentence_list = [] # Clear after speaking
        word_list = []
        last_input_time = time.time() # Reset to prevent spamming
    
    banner_height = 80
    cv2.rectangle(img, (0, h - banner_height), (w, h), COLOR_BG, -1)
    cv2.rectangle(img, (0, h - banner_height), (w, h - banner_height + 5), COLOR_ACCENT, -1)
    
    # if logo is not None:
    #     # Get the dimensions of the resized logo
    #     logo_h, logo_w = logo.shape[:2]
        
    #     # Calculate position (Middle Top with 10 pixels padding from the top edge)
    #     pad = 10
    #     y1, y2 = pad, pad + logo_h
        
    #     # Find the center of the screen, then offset by half the logo's width to center it
    #     x1 = (w // 2) - (logo_w // 2)
    #     x2 = x1 + logo_w

    #     # If the logo has a transparent (alpha) channel (4 channels: BGRA)
    #     if logo.shape[2] == 4:
    #         # Extract the alpha channel and create a mask
    #         alpha_logo = logo[:, :, 3] / 255.0
    #         alpha_img = 1.0 - alpha_logo

    #         # Blend the logo and the background image based on the alpha mask
    #         for c in range(0, 3):
    #             img[y1:y2, x1:x2, c] = (alpha_logo * logo[:, :, c] +
    #                                     alpha_img * img[y1:y2, x1:x2, c])
    #     else:
    #         # If the logo is a flat image (like a JPG), just paste it directly
    #         img[y1:y2, x1:x2] = logo

    font = cv2.FONT_HERSHEY_DUPLEX

    # Language indicator (top-left)
    lang_label = LANG_CONFIG[current_lang]["label"]
    lang_text = f"Lang: {lang_label}  [L]"
    font_small = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, lang_text, (15, 30), font_small, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, lang_text, (15, 30), font_small, 0.6, COLOR_ACCENT, 1, cv2.LINE_AA)

    display_text = full_sentence

    if len(display_text) > 0:
        cv2.putText(img, f"Input: {display_text}", (30, h - 110), font, 1.5, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, f"Input: {display_text}", (30, h - 110), font, 1.5, COLOR_ACCENT, 2, cv2.LINE_AA)
        
        # Dynamic Timer UI
        font_small = cv2.FONT_HERSHEY_SIMPLEX
        
        if pred_label != "Waiting...":
            # Hand is in frame
            timer_text = "Timer: PAUSED (Hand Detected)"
            timer_color = (0, 255, 0) # Green
        else:
            # Hand dropped, start counting
            time_left = max(0.0, TTS_DELAY - (time.time() - last_input_time))
            timer_text = f"Speaking in: {time_left:.1f}s"
            timer_color = (0, 200, 255) # Yellow

        cv2.putText(img, timer_text, (30, h - 160), font_small, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, timer_text, (30, h - 160), font_small, 0.7, timer_color, 2, cv2.LINE_AA)

    # Prediction text
    text = f"SIBI : {pred_label}"

    if has_appended:
        text += "   ADDED!"
        text_color = (0, 255, 0)
    else:
        text_color = COLOR_TEXT

    cv2.putText(img, text, (30, h - 25), font, 1.2, (0, 0, 0), 4, cv2.LINE_AA) # Shadow
    cv2.putText(img, text, (30, h - 25), font, 1.2, text_color, 2, cv2.LINE_AA) # Main text

    cv2.imshow("SIBI Gesture Classifier", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # Spacebar - finalize current word, start new one
        if len(word_list) > 0:
            sentence_list.append("".join(word_list))
            word_list = []
            last_input_time = time.time()
    elif key == ord('l') or key == ord('L'):  # Toggle language
        current_lang = "EN" if current_lang == "ID" else "ID"
        print(f"Language switched to: {LANG_CONFIG[current_lang]['label']}")
    elif key == 8 or key == 127:  # Backspace (Windows/Mac)
        if len(word_list) > 0:
            word_list.pop()
            last_input_time = time.time()
        elif len(sentence_list) > 0:
            # Pop last completed word back into word_list for editing
            word_list = list(sentence_list.pop())
            last_input_time = time.time()

cap.release()
cv2.destroyAllWindows()