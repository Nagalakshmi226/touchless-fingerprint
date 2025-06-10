import os
import cv2 as vision
import numpy as math
from keras.engine.training import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout
import pickle

DATA_PATH = '/content/drive/MyDrive/DB2_B'
IMG_SIZE = 100

image_stack = []
file_labels = []

for file in os.listdir(DATA_PATH):
    full_path = os.path.join(DATA_PATH, file)
    img = vision.imread(full_path, vision.IMREAD_GRAYSCALE)
    if img is not None:
        resized_img = vision.resize(img, (IMG_SIZE, IMG_SIZE))
        image_stack.append(resized_img)
        file_labels.append(file)

image_stack = math.array(image_stack).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = Conv2D(16, (5, 5), activation='swish')(input_layer)
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (3, 3), activation='swish')(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='swish')(x)
x = Dropout(0.3)(x)

custom_model = Model(inputs=input_layer, outputs=x)
custom_model.compile(optimizer='nadam', loss='mae')

embeddings = custom_model.predict(image_stack)

with open('/content/drive/MyDrive/fingerprint_embeds.pkl', 'wb') as f:
    pickle.dump({'vectors': embeddings, 'labels': file_labels}, f)

custom_model.save('/content/drive/MyDrive/fingerprint_model.keras')
