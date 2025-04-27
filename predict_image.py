import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")

model = load_model(MODEL_PATH)

# تحميل أسماء الكلاسات
with open("class_names.json") as f:
    class_indices = json.load(f)
class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_disease(img_path):
    processed_img = prepare_image(img_path)
    prediction = model.predict(processed_img)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class
