import tensorflow as tf
from keras.models import load_model

model = load_model("plant_model.h5", compile=False)

print("Model loaded successfully")