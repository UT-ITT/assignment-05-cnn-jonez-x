"""
Annotation tool: click top-left then bottom-right corner of the hand.
Press 'r' to redo the current image, 's' to skip, 'q' to quit and save.
"""
import cv2
import json
import os

IMAGE_DIR = './my_images'
OUTPUT_FILE = './annot-jonez.json'

# known labels per file
LABELS = {
    'like_01': 'like',
    'like_02': 'like',
    'like_03': 'like',
    'peace_01': 'peace',
    'peace_02': 'peace',
    'peace_03': 'peace',
    'rock_01': 'rock',
    'rock_02': 'rock',
    'rock_03': 'rock',
}

annotations = {}
clicks = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))

def annotate():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            annotations.update(json.load(f))

    files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')])

    for filename in files:
        key = filename.lower().replace('.jpg', '').replace('.jpeg', '')
        if key in annotations:
            print(f'  Skipping {filename} (already annotated)')
            continue
        label = LABELS.get(key, 'unknown')
        path = os.path.join(IMAGE_DIR, filename)

        img_orig = cv2.imread(path)
        h_orig, w_orig = img_orig.shape[:2]

        # scale down for display (max 1200px wide)
        scale = min(1200 / w_orig, 900 / h_orig)
        display_w = int(w_orig * scale)
        display_h = int(h_orig * scale)
        img_display = cv2.resize(img_orig, (display_w, display_h))

        cv2.namedWindow(filename)
        cv2.setMouseCallback(filename, on_mouse)

        while True:
            clicks.clear()
            img_copy = img_display.copy()
            cv2.putText(img_copy, f'{filename} | label: {label} | click top-left then bottom-right | r=redo s=skip q=quit',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow(filename, img_copy)

            # wait for 2 clicks
            while len(clicks) < 2:
                key_pressed = cv2.waitKey(20) & 0xFF
                if key_pressed == ord('s'):
                    cv2.destroyWindow(filename)
                    break
                if key_pressed == ord('q'):
                    cv2.destroyAllWindows()
                    save(annotations)
                    return
                img_copy2 = img_display.copy()
                cv2.putText(img_copy2, f'{filename} | label: {label} | clicks: {len(clicks)}/2',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                for c in clicks:
                    cv2.circle(img_copy2, c, 6, (0, 0, 255), -1)
                cv2.imshow(filename, img_copy2)
            else:
                # draw rectangle for confirmation
                x1d, y1d = clicks[0]
                x2d, y2d = clicks[1]
                img_confirm = img_display.copy()
                cv2.rectangle(img_confirm, (x1d, y1d), (x2d, y2d), (0, 255, 0), 2)
                cv2.putText(img_confirm, 'Press ENTER to confirm, r to redo',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow(filename, img_confirm)

                k = cv2.waitKey(0) & 0xFF
                if k == 13:  # Enter
                    # convert display coords back to original image coords, then normalize
                    x1 = min(clicks[0][0], clicks[1][0]) / scale
                    y1 = min(clicks[0][1], clicks[1][1]) / scale
                    x2 = max(clicks[0][0], clicks[1][0]) / scale
                    y2 = max(clicks[0][1], clicks[1][1]) / scale

                    bx = x1 / w_orig
                    by = y1 / h_orig
                    bw = (x2 - x1) / w_orig
                    bh = (y2 - y1) / h_orig

                    annotations[key] = {
                        'bboxes': [[round(bx, 8), round(by, 8), round(bw, 8), round(bh, 8)]],
                        'labels': [label],
                        'landmarks': [],
                        'leading_conf': 1.0,
                        'leading_hand': 'right',
                        'user_id': ''
                    }
                    print(f'  Saved: {key} -> {label} bbox=[{bx:.4f}, {by:.4f}, {bw:.4f}, {bh:.4f}]')
                    cv2.destroyWindow(filename)
                    break
                elif k == ord('r'):
                    continue  # redo
                continue

    cv2.destroyAllWindows()
    save(annotations)

def save(data):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'\nDone! Saved {len(data)} annotations to {OUTPUT_FILE}')

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    annotate()
