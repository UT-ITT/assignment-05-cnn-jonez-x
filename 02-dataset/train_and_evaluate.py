"""
Trains a CNN on the HaGRID subset (like, peace, rock),
then runs predictions on own images + tutor images and saves conf-matrix.png.
"""
import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, RandomFlip, RandomContrast
from keras.metrics import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# --- Config ---
CONDITIONS = ['like', 'peace', 'rock']
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)
COLOR_CHANNELS = 3
DATASET_PATH = '../gesture_dataset_sample'
MY_IMAGES_PATH = './my_images'
TUTOR_IMAGES_PATH = './tutor_images'
MY_ANNOT_FILE = './annot-jonez.json'
TUTOR_ANNOT_FILE = './tutor_images/annot-tutors.json'
MODEL_PATH = './gesture_recognition.keras'

def preprocess_image(img):
    img_resized = cv2.resize(img, SIZE)
    return img_resized

# --- Load HaGRID training data ---
print('Loading HaGRID dataset...')
annotations = {}
for condition in CONDITIONS:
    with open(f'{DATASET_PATH}/_annotations/{condition}.json') as f:
        annotations[condition] = json.load(f)

images, labels, label_names = [], [], []

for condition in CONDITIONS:
    img_dir = f'{DATASET_PATH}/{condition}'
    if not os.path.exists(img_dir):
        print(f'  Warning: {img_dir} not found, skipping')
        continue
    for filename in tqdm(os.listdir(img_dir), desc=condition):
        UID = filename.split('.')[0]
        img = cv2.imread(f'{img_dir}/{filename}')
        if img is None:
            continue
        try:
            annotation = annotations[condition][UID]
        except KeyError:
            continue
        for i, bbox in enumerate(annotation['bboxes']):
            label = annotation['labels'][i]
            if label not in CONDITIONS:
                continue
            x1 = int(bbox[0] * img.shape[1])
            y1 = int(bbox[1] * img.shape[0])
            w  = int(bbox[2] * img.shape[1])
            h  = int(bbox[3] * img.shape[0])
            crop = img[y1:y1+h, x1:x1+w]
            if crop.size == 0:
                continue
            preprocessed = preprocess_image(crop)
            if label not in label_names:
                label_names.append(label)
            images.append(preprocessed)
            labels.append(label_names.index(label))

print(f'Loaded {len(images)} images, classes: {label_names}')

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = np.array(X_train).astype('float32') / 255.
X_test  = np.array(X_test).astype('float32')  / 255.
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
X_test  = X_test.reshape(-1,  IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)

y_train_oh = to_categorical(y_train, num_classes=len(label_names))
y_test_oh  = to_categorical(y_test,  num_classes=len(label_names))

# --- Build model ---
print('Building model...')
model = Sequential([
    RandomFlip('horizontal'),
    RandomContrast(0.1),
    Conv2D(64, (9, 9), activation='leaky_relu', input_shape=(IMG_SIZE, IMG_SIZE, COLOR_CHANNELS), padding='same'),
    MaxPooling2D((4, 4), padding='same'),
    Conv2D(32, (5, 5), activation='leaky_relu', padding='same'),
    MaxPooling2D((3, 3), padding='same'),
    Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(label_names), activation='softmax'),
])

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
stop_early = EarlyStopping(monitor='val_loss', patience=3)

# --- Train ---
print('Training...')
model.fit(
    X_train, y_train_oh,
    batch_size=32,
    epochs=50,
    verbose=1,
    validation_data=(X_test, y_test_oh),
    callbacks=[reduce_lr, stop_early]
)

model.save(MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')

# --- Helper: predict on a set of images from an annotation file ---
def predict_from_annot(annot_file, image_dir):
    with open(annot_file) as f:
        annot = json.load(f)

    preds, truths = [], []
    for key, entry in annot.items():
        # find image file (try jpg/JPG/png)
        img_path = None
        for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
            p = os.path.join(image_dir, key + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            print(f'  Image not found: {key}')
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        for i, bbox in enumerate(entry['bboxes']):
            label = entry['labels'][i]
            if label not in label_names:
                continue
            x1 = int(bbox[0] * img.shape[1])
            y1 = int(bbox[1] * img.shape[0])
            w  = int(bbox[2] * img.shape[1])
            h  = int(bbox[3] * img.shape[0])
            crop = img[y1:y1+h, x1:x1+w]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, SIZE).astype('float32') / 255.
            crop = crop.reshape(1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
            pred = model.predict(crop, verbose=0)
            preds.append(label_names[np.argmax(pred)])
            truths.append(label)

    return truths, preds

# --- Predict on own + tutor images ---
print('\nRunning predictions on own images...')
truths_mine, preds_mine = predict_from_annot(MY_ANNOT_FILE, MY_IMAGES_PATH)

print('Running predictions on tutor images...')
truths_tutor, preds_tutor = predict_from_annot(TUTOR_ANNOT_FILE, TUTOR_IMAGES_PATH)

all_truths = truths_mine + truths_tutor
all_preds  = preds_mine  + preds_tutor

print(f'\nPredictions: {list(zip(all_truths, all_preds))}')

# --- Confusion matrix ---
cm = confusion_matrix(all_truths, all_preds, labels=label_names)
fig, ax = plt.subplots(figsize=(7, 7))
ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax)
ax.set_title('Confusion Matrix — own + tutor images')
plt.tight_layout()
plt.savefig('./conf-matrix.png', dpi=150)
print('Saved conf-matrix.png')
plt.show()
