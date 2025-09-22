import tensorflow as tf
print(tf.__version__)


# import joblib

# pca = joblib.load('model/PCA_ECG.pkl')
# model = joblib.load('model/Heart_Disease_Prediction_using_ECG.pkl')

# print(type(pca))
# print(type(model))


from tensorflow.keras.models import load_model

model = load_model("./model/brain_tumor_model.h5")

# Print general info
print(model.summary())

# Check optimizer config (often includes version-related details)
if model.optimizer:
    print("Optimizer:", model.optimizer.get_config())


import h5py

with h5py.File("./model/brain_tumor_model.h5", "r") as f:
    print("Keras version:", f.attrs.get("keras_version"))
    print("Backend:", f.attrs.get("backend"))
