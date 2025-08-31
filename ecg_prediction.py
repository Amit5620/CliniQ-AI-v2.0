"""
ECG Prediction Module
Handles ECG image processing and prediction functionality
"""

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import request, flash, redirect, render_template, session
from models import db, Prediction
from Ecg import ECG

def handle_ecg_prediction(user):
    """Handle ECG prediction with comprehensive processing"""
    try:
        from roboflow import Roboflow
        import supervision as sv
        from flask import current_app
        
        # Get form data from session
        ecg_form_data = session.get('ecg_form_data', {})
        file = ecg_form_data.get('file')
        if not file:
            file = request.files.get('file')
        
        if not file:
            flash('Please upload an ECG image', 'error')
            return redirect(request.referrer or '/ecg_predict')
        
        filename = secure_filename(file.filename)
        filename = f"ecg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        file_path = os.path.join('static/uploads', filename)
        
        # Ensure upload directory exists
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        file.save(file_path)
        
        # Get patient details
        from flask_login import current_user
        
        patient_name = ecg_form_data.get('patient_name') or request.form.get('patient_name')
        patient_age = int(ecg_form_data.get('patient_age') or request.form.get('patient_age'))
        patient_gender = ecg_form_data.get('patient_gender') or request.form.get('patient_gender')
        patient_id = f"ECG-{current_user.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        doctor_name = ecg_form_data.get('doctor_name') or request.form.get('doctor_name')
        doctor_type = ecg_form_data.get('doctor_type') or request.form.get('doctor_type')
        symptoms = ecg_form_data.get('patient_symptoms') or request.form.get('patient_symptoms')
        disease_type = ecg_form_data.get('disease_type') or 'ecg'
        
        # Ensure process folder exists
        process_folder = os.path.join('static', 'process')
        if not os.path.exists(process_folder):
            os.makedirs(process_folder)
        
        # Process with ECG class
        ecg = ECG()
        user_image = ecg.getImage(file_path)
        gray_image = ecg.GrayImgae(user_image)
        gray_path = os.path.join(process_folder, 'gray_image.png')
        cv2.imwrite(gray_path, gray_image * 255)

        leads = ecg.DividingLeads(user_image)
        ecg.PreprocessingLeads(leads)
        ecg.SignalExtraction_Scaling(leads)

        ecg_1d_signal = ecg.CombineConvert1Dsignal()
        ecg_reduced = ecg.DimensionalReduciton(ecg_1d_signal)
        ecg_prediction_result = ecg.ModelLoad_predict(ecg_reduced)

        if ecg_prediction_result == 'Your ECG is Normal':
            ecg_prediction = str(ecg_prediction_result) + ' [ Normal ]'
        else:
            ecg_prediction = str(ecg_prediction_result)

        # Save CSV files
        pd.DataFrame(ecg_1d_signal).to_csv(os.path.join(process_folder, '1d_signal.csv'), index=False)
        pd.DataFrame(ecg_reduced).to_csv(os.path.join(process_folder, 'reduced_signal.csv'), index=False)

        # Roboflow Models
        # rf = Roboflow(api_key="kqlgTsdyBapHPYnoxznG")
        rf = Roboflow(api_key="onkABfIuwzoDBGtprD7B")
        model1 = rf.workspace().project("ecg-classification-ygs4v").version(1).model
        model2 = rf.workspace().project("ecg_detection").version(3).model

        result1 = model1.predict(file_path, confidence=40, overlap=30).json()
        result2 = model2.predict(file_path, confidence=40, overlap=30).json()

        base_name, _ = os.path.splitext(os.path.basename(filename))


        def process_result(result, suffix):
            predictions = result.get("predictions", [])
            if not predictions:
                # return filename, ["No abnormality detected"]
                return f"{base_name}_{suffix}.jpg", ["No abnormality detected"]


            xyxy, confidence, class_id, labels = [], [], [], []
            predicted_classes = []

            for pred in predictions:
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                xyxy.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
                confidence.append(pred["confidence"])
                class_id.append(pred["class_id"])
                labels.append(pred["class"])
                predicted_classes.append(pred["class"])

            xyxy = np.array(xyxy, dtype=np.float32)
            confidence = np.array(confidence, dtype=np.float32)
            class_id = np.array(class_id, dtype=int)

            detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            img = cv2.imread(file_path)
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated = box_annotator.annotate(scene=img, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            # output_path = os.path.join(process_folder, filename)
            # cv2.imwrite(output_path, annotated)
            # return filename, predicted_classes
            output_filename = f"{base_name}_{suffix}.jpg"
            output_path = os.path.join(process_folder, output_filename)
            cv2.imwrite(output_path, annotated)
            return output_filename, predicted_classes

        pred_img1, predicted_classes1 = process_result(result1, 'prediction1.jpg')
        pred_img2, predicted_classes2 = process_result(result2, 'prediction2.jpg')

        # Generate ECG processing outputs
        ecg_outputs = {
            'gray': 'gray_image.png',
            'lead_12': 'Leads_1-12_figure.png',
            'lead_13': 'Long_Lead_13_figure.png',
            'preprocessed_12': 'Preprossed_Leads_1-12_figure.png',
            'preprocessed_13': 'Preprossed_Leads_13_figure.png',
            'contour': 'Contour_Leads_1-12_figure.png'
        }

        # Save prediction to DB
        from flask_login import current_user
        
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
            prediction_result=str(ecg_prediction),
        )
        
        db.session.add(new_prediction)
        db.session.commit()
        
        # Prepare data for template
        return {
            'original': filename,
            'pred1': pred_img1,
            'pred2': pred_img2,
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'patient_id': patient_id,
            'doctor_name': doctor_name,
            'doctor_type': doctor_type,
            'disease_type': disease_type,
            'patient_symptoms': symptoms,
            'predicted_classes1': predicted_classes1,
            'predicted_classes2': predicted_classes2,
            'ecg_prediction': ecg_prediction,
            'ecg_outputs': ecg_outputs,
            'ecg_1d_signal': pd.DataFrame(ecg_1d_signal),
            'ecg_reduced': pd.DataFrame(ecg_reduced)
        }
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error processing ECG: {str(e)}', 'error')
        return None
