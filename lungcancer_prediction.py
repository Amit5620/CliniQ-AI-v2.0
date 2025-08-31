"""
Lung Cancera Prediction Module
Handles Lung Cancer image processing and prediction functionality
"""

import os
import cv2
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import request, flash, redirect, render_template, session
from models import db, Prediction

from PIL import Image
import cv2
import matplotlib as mpl
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.image as img
from tensorflow.keras.models import load_model

last_conv_layer_name = "Top_Conv_Layer"

def get_img_array(img, size = (224 , 224)):
    image = np.array(img)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    resized_image = cv2.resize(image, (224,224))
    resized_image = resized_image.reshape(-1,224,224,3)
    resized_image = np.array(resized_image)
    return resized_image
    # img = keras.utils.load_img(img_path, target_size=size)
    # array = keras.utils.img_to_array(img)
    # array = np.expand_dims(array, axis=0)
    # return array


def make_gradcam_heatmap(img_array, model , last_conv_layer_name = last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4 , view = False):
    # Load the original image
    img = np.array(img)
    # img = keras.utils.load_img(img_path)
    # img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    if view :
        display(Image(cam_path))

        
     
def decode_predictions(preds):
    classes = ['Bengin Case', 'Malignant Case', 'Normal Case']
    prediction = classes[np.argmax(preds)]
    return prediction



def make_prediction (img , model, last_conv_layer_name = last_conv_layer_name , campath = "cam.jpeg" , view = False):
    image = get_img_array(img)
    img_array = get_img_array(img, size=(224 , 224))
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img, heatmap , cam_path=campath , view = view)
    return [campath , decode_predictions(preds)]


def handle_lungcancer_prediction(user):
    """Handle Lung Cancer prediction with comprehensive processing"""
    last_conv_layer_name = "Top_Conv_Layer"

    try:
        from flask import current_app
        
        # Get form data from session
        lungcancer_form_data = session.get('lungcancer_form_data', {})
        file = lungcancer_form_data.get('file')
        if not file:
            file = request.files.get('file')
        
        if not file:
            flash('Please upload an Lung Cancer image', 'error')
            return redirect(request.referrer or '/lungcancer_predict')
        
        filename = secure_filename(file.filename)
        filename = f"lungcancer_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        file_path = os.path.join('static/uploads', filename)
        
        # Ensure upload directory exists
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        file.save(file_path)
        
        patient_name = lungcancer_form_data.get('patient_name') or request.form.get('patient_name')
        patient_age = int(lungcancer_form_data.get('patient_age') or request.form.get('patient_age'))
        patient_gender = lungcancer_form_data.get('patient_gender') or request.form.get('patient_gender')
        patient_id = f"LC-{user}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        doctor_name = lungcancer_form_data.get('doctor_name') or request.form.get('doctor_name')
        doctor_type = lungcancer_form_data.get('doctor_type') or request.form.get('doctor_type')
        symptoms = lungcancer_form_data.get('patient_symptoms') or request.form.get('patient_symptoms')
        disease_type = lungcancer_form_data.get('disease_type') or 'Lung Cancer'

        print(patient_name, patient_age, patient_gender, patient_id, disease_type)
        print(file)
        
        # Ensure process folder exists
        process_folder = os.path.join('static', 'process')
        if not os.path.exists(process_folder):
            os.makedirs(process_folder)
        
        # Process with Lung Cancer class

        # Load the Model
        image = Image.open(file_path)
        model = load_model("./model/lung_cancer_model.h5")

        base_name, _ = os.path.splitext(os.path.basename(filename))

        campath = f"{base_name}_gradcam.jpg"
        campath_full = os.path.join(process_folder, campath)
        _, result_model = make_prediction(image, model, campath=campath_full, view=False)

        if( result_model == "Normal case"):
            result = f"{result_model} Detected [ Normal ]"
        else:
            result = f"{result_model} Detected"

        def process_result(result, suffix):
            # result is a string prediction, not a dict
            if not result or result == "":
                return f"{base_name}_{suffix}.jpg", ["No abnormality detected"]

            # For  lungcancer, the predicted image is the gradcam
            output_filename = f"{base_name}_{suffix}.jpg"
            output_path = os.path.join(process_folder, output_filename)
            # Copy the gradcam to the predicted image path
            shutil.copy(campath_full, output_path)
            return output_filename, [result]

        pred_img1, predicted_classes = process_result(result, 'prediction.jpg')





        # Save prediction to DB
        new_prediction = Prediction(
            user_id=user,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            patient_id=patient_id,
            referring_doctor=doctor_name,
            symptoms=symptoms,
            disease_type=disease_type,
            original_image=file_path,
            predicted_image=os.path.join('static/process', pred_img1),
            prediction_result=str(result),
        )
        
        db.session.add(new_prediction)
        db.session.commit()
        
        # Prepare data for template
        return {
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'patient_id': patient_id,
            'doctor_name': doctor_name,
            'doctor_type': doctor_type,
            'disease_type': disease_type,
            'patient_symptoms': symptoms,
            'prediction_result': result,
            'original_image': filename,
            'predicted_image': pred_img1
        }
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error processing Lung Cancer: {str(e)}', 'error')
        return None
