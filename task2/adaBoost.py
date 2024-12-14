
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Configuration parameters
CONFIG = {
    "img_size": (128, 128),
    "batch_size": 32,
    "epochs": 10,
    "train_path": 'Alzheimer_s Dataset/Alzheimer_s Dataset/train',
    "validation_path": 'Alzheimer_s Dataset/Alzheimer_s Dataset/test',
    "cnn_model_path": 'cnn_model.h5',
    "ada_model_path": 'ada_classifier_model.pkl',
    "class_labels": {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'},
}

# Data augmentation
def create_datagen(rescale, augment=False):
    if augment:
        return ImageDataGenerator(
            rescale=rescale,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    return ImageDataGenerator(rescale=rescale)

train_gen = create_datagen(1.0 / 255, augment=True).flow_from_directory(
    CONFIG["train_path"],
    target_size=CONFIG["img_size"],
    batch_size=CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=False
)

val_gen = create_datagen(1.0 / 255).flow_from_directory(
    CONFIG["validation_path"],
    target_size=CONFIG["img_size"],
    batch_size=CONFIG["batch_size"],
    class_mode='categorical',
    shuffle=False
)

# CNN Model definition
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train or load CNN model
if os.path.exists(CONFIG["cnn_model_path"]):
    print("Loading existing CNN model...")
    cnn_model = load_model(CONFIG["cnn_model_path"])
else:
    print("Training new CNN model...")
    cnn_model = build_cnn_model()
    cnn_model.fit(train_gen, epochs=CONFIG["epochs"], validation_data=val_gen)
    cnn_model.save(CONFIG["cnn_model_path"])
    print("CNN model saved.")

# Feature extraction
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

def extract_features(model, data_gen):
    features, labels = [], []
    for _ in range(len(data_gen)):
        imgs, lbls = next(data_gen)
        features.append(model.predict(imgs))
        labels.extend(lbls)
    return np.vstack(features), np.array(labels)

train_features, train_labels = extract_features(feature_extractor, train_gen)
val_features, val_labels = extract_features(feature_extractor, val_gen)
val_labels = np.argmax(val_labels, axis=1)

# Train or load AdaBoost classifier
if os.path.exists(CONFIG["ada_model_path"]):
    print("Loading existing AdaBoost model...")
    ada_model = joblib.load(CONFIG["ada_model_path"])
else:
    print("Training new AdaBoost model...")
    ada_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50)
    ada_model.fit(train_features, np.argmax(train_labels, axis=1))
    joblib.dump(ada_model, CONFIG["ada_model_path"])
    print("AdaBoost model saved.")

# Evaluate AdaBoost model
val_predictions = ada_model.predict(val_features)
accuracy = accuracy_score(val_labels, val_predictions)
print(f"AdaBoost Model Accuracy: {accuracy:.2%}")

# Image classification
def classify_image(img_path):
    img = load_img(img_path, target_size=CONFIG["img_size"])
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array)
    prediction = ada_model.predict(features)
    return CONFIG["class_labels"][prediction[0]]

# Test the classifier
test_image_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test/NonDemented/26 (66).jpg'
result = classify_image(test_image_path)
print(f"Predicted Class: {result}")