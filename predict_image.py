import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

model = None
class_names = None

def load_model_and_classes():
    global model, class_names
    if model is None:
        print("Loading model and class names...")
        if not os.path.exists("best_model.keras"):
            raise FileNotFoundError("ERROR: Model file not found!")
        model = load_model("best_model.keras")
        
        with open("class_names.json") as f:
            class_indices = json.load(f)
        class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
        print("Model and classes loaded.")

def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_disease(img_path):
    load_model_and_classes()
    processed_img = prepare_image(img_path)
    prediction = model.predict(processed_img)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class
