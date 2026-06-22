"""
Gesture-based Media Controller
- like    -> play/pause
- peace   -> skip track
- rock    -> increase volume
- no gesture (low confidence) -> no action

Controls are sent via pynput (macOS media keys).
Press 'q' in the OpenCV window to quit.
"""
import cv2
import numpy as np
import time
import keras
from pynput.keyboard import Key, Controller

# --- Config ---
MODEL_PATH = '../02-dataset/gesture_recognition.keras'
IMG_SIZE = 64
COLOR_CHANNELS = 3
CONFIDENCE_THRESHOLD = 0.97   # must be this confident to count
HOLD_SECONDS = 5.0            # must hold gesture this long before action fires
COOLDOWN_SECONDS = 2.0        # minimum time between two actions

# Gesture -> media key mapping
GESTURE_ACTIONS = {
    'like':  'play_pause',
    'peace': 'next_track',
    'rock':  'volume_up',
}

keyboard = Controller()

def send_media_key(action):
    if action == 'play_pause':
        keyboard.press(Key.media_play_pause)
        keyboard.release(Key.media_play_pause)
    elif action == 'next_track':
        keyboard.press(Key.media_next)
        keyboard.release(Key.media_next)
    elif action == 'volume_up':
        keyboard.press(Key.media_volume_up)
        keyboard.release(Key.media_volume_up)
        keyboard.press(Key.media_volume_up)
        keyboard.release(Key.media_volume_up)

def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.
    return img.reshape(1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)

def draw_overlay(frame, gesture, confidence, hold_progress, action, on_cooldown):
    h, w = frame.shape[:2]

    # draw center crop box
    cx, cy = w // 2, h // 2
    box_size = min(w, h) // 2
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2
    box_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # gesture + confidence
    cv2.putText(frame, f'{gesture} ({confidence:.0%})',
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2)

    # hold progress bar
    if hold_progress > 0:
        bar_w = int((w - 20) * hold_progress)
        cv2.rectangle(frame, (10, 55), (w - 10, 75), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 55), (10 + bar_w, 75), (0, 220, 0), -1)
        cv2.putText(frame, f'Hold... {hold_progress*HOLD_SECONDS:.1f}s / {HOLD_SECONDS:.0f}s',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # action feedback
    if action and on_cooldown:
        cv2.putText(frame, f'-> {action}',
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, 'q = quit',
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return frame

def main():
    print('Loading model...')
    model = keras.models.load_model(MODEL_PATH)
    label_names = ['like', 'peace', 'rock']  # must match training order

    print('Starting camera...')
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)  # let camera warm up on macOS

    last_action_time = 0
    last_action = ''
    gesture_start_time = None  # when current confident gesture started
    current_hold_gesture = None

    print('Ready — hold a gesture inside the green box for 5 seconds. Press q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # crop center region for prediction
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        box_size = min(w, h) // 2
        x1, y1 = cx - box_size // 2, cy - box_size // 2
        x2, y2 = cx + box_size // 2, cy + box_size // 2
        crop = frame[y1:y2, x1:x2]

        inp = preprocess(crop)
        preds = model.predict(inp, verbose=0)[0]
        confidence = float(np.max(preds))
        gesture = label_names[int(np.argmax(preds))]

        now = time.time()
        on_cooldown = (now - last_action_time) < COOLDOWN_SECONDS

        # track how long the same gesture is held above threshold
        if confidence >= CONFIDENCE_THRESHOLD and not on_cooldown:
            if gesture == current_hold_gesture:
                hold_duration = now - gesture_start_time
            else:
                current_hold_gesture = gesture
                gesture_start_time = now
                hold_duration = 0.0
        else:
            current_hold_gesture = None
            gesture_start_time = None
            hold_duration = 0.0

        hold_progress = min(hold_duration / HOLD_SECONDS, 1.0) if hold_duration > 0 else 0.0

        # fire action when held long enough
        if hold_progress >= 1.0:
            action = GESTURE_ACTIONS.get(gesture, '')
            if action:
                send_media_key(action)
                last_action_time = now
                last_action = action
                current_hold_gesture = None
                gesture_start_time = None
                print(f'[{gesture}] -> {action} ({confidence:.0%})')
        else:
            action = ''

        display = draw_overlay(frame.copy(), gesture, confidence, hold_progress,
                               last_action if on_cooldown else action,
                               on_cooldown)
        cv2.imshow('Gesture Media Control', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
