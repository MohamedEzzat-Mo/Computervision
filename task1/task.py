import os
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CONFIG = {
    "img_size": (128, 128),
    "batch_size": 32,
    "epochs": 20,
    "train_path": "dataset/train",
    "validation_path": "dataset/validation",
    "cnn_model_path": "cnn_model.h5",
    "knn_model_path": "knn_model.pkl",
}

def create_data_generators(config):
    """Creates training and validation data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_datagen.flow_from_directory(
        config["train_path"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="binary",
    )

    validation_data = validation_datagen.flow_from_directory(
        config["validation_path"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="binary",
    )

    return train_data, validation_data

def build_cnn_model(input_shape):
    """Builds and compiles the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation="relu"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def extract_features(model, data):
    """Extracts features using a trained CNN model."""
    features, labels = [], []
    for _ in range(len(data)):
        imgs, lbls = next(data)
        features.extend(model.predict(imgs))
        labels.extend(lbls)
    return np.array(features), np.array(labels)

def train_or_load_cnn(config, train_data, validation_data):
    """Trains or loads a CNN model based on existence."""
    if os.path.exists(config["cnn_model_path"]):
        print("Loading CNN model...")
        return load_model(config["cnn_model_path"])

    print("Training CNN model...")
    model = build_cnn_model((*config["img_size"], 3))
    model.fit(train_data, epochs=config["epochs"], validation_data=validation_data)
    model.save(config["cnn_model_path"])
    print("CNN model saved.")
    return model

def train_or_load_knn(config, train_features, train_labels):
    """Trains or loads a KNN model based on existence."""
    if os.path.exists(config["knn_model_path"]):
        print("Loading KNN model...")
        return joblib.load(config["knn_model_path"])

    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=11, metric="euclidean")
    knn.fit(train_features, train_labels)
    joblib.dump(knn, config["knn_model_path"])
    print("KNN model saved.")
    return knn

def evaluate_model(knn, validation_features, validation_labels):
    """Evaluates the KNN model."""
    predictions = knn.predict(validation_features)
    accuracy = accuracy_score(validation_labels, predictions)
    print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")
    return predictions

def display_predictions(validation_data, random_indices, predictions):
    """Displays predictions for random samples."""
    validation_data.reset()
    sample_images, real_labels = [], []
    for idx in random_indices:
        imgs, lbls = next(validation_data)
        sample_images.append(imgs[idx % CONFIG["batch_size"]])
        real_labels.append(lbls[idx % CONFIG["batch_size"]])

    plt.figure(figsize=(12, 8))
    for i, (image, real, pred) in enumerate(zip(sample_images, real_labels, predictions)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.title(f"Real: {real}\nPred: {pred}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Main workflow
train_data, validation_data = create_data_generators(CONFIG)
cnn_model = train_or_load_cnn(CONFIG, train_data, validation_data)
train_features, train_labels = extract_features(cnn_model, train_data)
validation_features, validation_labels = extract_features(cnn_model, validation_data)

knn_model = train_or_load_knn(CONFIG, train_features, train_labels)
predictions = evaluate_model(knn_model, validation_features, validation_labels)

random_indices = random.sample(range(len(validation_labels)), 10)
display_predictions(validation_data, random_indices, predictions)