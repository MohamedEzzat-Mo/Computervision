#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, Activation, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image

# Configuration
CONFIG = {
    "input_shape": (1024, 1024),
    "model_path": "U-net_model1.h5",
    "image_path": "loli.png",
    "num_classes": 2,  # Modify as needed for segmentation classes
}

# Encoder Block: Conv2D -> ReLU -> Conv2D -> ReLU -> MaxPooling
def encoder_block(input_tensor, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = Activation("relu")(x)
    p = MaxPooling2D((2, 2))(x)
    return x, p

# Decoder Block: UpSampling2D -> Conv2D -> Concatenate -> Conv2D -> ReLU
def decoder_block(input_tensor, skip_tensor, num_filters):
    x = UpSampling2D((2, 2))(input_tensor)
    x = Conv2D(num_filters, 2, padding="same")(x)
    x = Concatenate()([x, skip_tensor])
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = Activation("relu")(x)
    return x

# U-Net Model
def unet_model(input_shape=(256, 256, 3), num_classes=2):
    inputs = Input(input_shape)

    # Contracting Path (Encoder)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    s5, p5 = encoder_block(p4, 1024)

    # Bottleneck
    b1 = Conv2D(2048, 3, padding="same")(p5)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(2048, 3, padding="same")(b1)
    b1 = Activation("relu")(b1)

    # Expansive Path (Decoder)
    d0 = decoder_block(b1, s5, 1024)
    d1 = decoder_block(d0, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output Layer
    outputs = Conv2D(num_classes, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net")
    return model

# Load or train the U-Net model
def load_or_train_unet(model_path, input_shape, num_classes):
    if os.path.exists(model_path):
        print("Loading pre-trained U-Net model...")
        model = load_model(model_path)
    else:
        print("Training U-Net model...")
        model = unet_model(input_shape=input_shape.__add__((3,)), num_classes=num_classes)
        model.save(model_path)
        print("U-Net model saved.")
    return model

# Preprocess Image
def preprocess_image(img_path, input_shape):
    img = Image.open(img_path).resize(input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array[:, :, :3], axis=0) / 255.0
    return img, img_array

# Post-process the Prediction
def postprocess_prediction(pred, original_img):
    pred = np.squeeze(pred, axis=0)  # Remove batch dimension
    pred = np.argmax(pred, axis=-1)  # If num_classes > 1, remove channel dimension
    pred_img = Image.fromarray(np.uint8(pred * 255))  # Convert to grayscale image  
    pred_img = pred_img.resize((original_img.width, original_img.height))  # Resize back to original dimensions
    return pred_img

# Visualization
def visualize_prediction(pred_img):
    plt.imshow(pred_img, cmap='gray')
    plt.axis('off')
    plt.show()

# Main Workflow
def main():
    model = load_or_train_unet(CONFIG["model_path"], CONFIG["input_shape"], CONFIG["num_classes"])
    img, img_array = preprocess_image(CONFIG["image_path"], CONFIG["input_shape"])
    pred = model.predict(img_array)
    pred_img = postprocess_prediction(pred, img)
    visualize_prediction(pred_img)

if __name__ == "__main__":
    main()