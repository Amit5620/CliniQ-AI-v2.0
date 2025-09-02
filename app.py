# CliniQ AI - A Comprehensive Health Prediction and Management System

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_security import Security, SQLAlchemyUserDatastore, roles_required, roles_accepted, RoleMixin
from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from email_validator import validate_email, EmailNotValidError
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import json
import secrets
from functools import wraps
import uuid
import pandas as pd

# Import configuration
from config import config

# Import ML models
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Import custom modules
from models import db, User, Role, Prediction, Appointment, ContactMessage, roles_users
from Ecg import ECG
from ecg_prediction import handle_ecg_prediction
from braintumor_prediction import handle_braintumor_prediction
from pneumonia_prediction import handle_pneumonia_prediction
from lungcancer_prediction import handle_lungcancer_prediction

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'default')
app.config.from_object(config[config_name])

# Ensure directories exist
config[config_name].init_app(app)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
Session(app)

# Add datetime.now to Jinja2 globals
app.jinja_env.globals['now'] = datetime.now

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load ML models
def load_ml_models():
    """Load all ML models for disease prediction"""
    try:
        global brain_tumor_model, pneumonia_model, lung_cancer_model
        brain_tumor_model = load_model('model/brain_tumor_model.h5')
        pneumonia_model = load_model('model/pneumonia_model.h5')
        lung_cancer_model = load_model('model/lung_cancer_model.h5')
        print("ML models loaded successfully")
    except Exception as e:
        print(f"Error loading ML models: {e}")

# Validation functions
def validate_password(password):
    """Validate password contains only letters and numbers"""
    return password.isalnum() and len(password) >= 6

def validate_gmail(email):
    """Validate email ends with @gmail.com"""
    try:
        valid = validate_email(email)
        return valid.email.endswith('@gmail.com')
    except EmailNotValidError:
        return False

def validate_phone(phone):
    """Validate phone number format"""
    import re
    # Remove spaces and check if it's a valid Indian phone number
    phone = phone.replace(" ", "")
    pattern = r'^[\+]?[1-9][\d]{9,12}$'
    return bool(re.match(pattern, phone))

# Decorators
def role_required(role):
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if not current_user.has_role(role):
                flash('Access denied. Insufficient permissions.', 'error')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Routes
@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact page with form submission"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        if not all([name, email, subject, message]):
            flash('Please fill all fields', 'error')
            return redirect(url_for('contact'))
        
        contact_msg = ContactMessage(
            name=name,
            email=email,
            subject=subject,
            message=message
        )
        db.session.add(contact_msg)
        db.session.commit()
        
        flash('Your message has been sent successfully!', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    """Redirect to user registration"""
    return redirect(url_for('register_user'))

@app.route('/register/user', methods=['GET', 'POST'])
def register_user():
    """User registration route"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        
        # Validation
        if not all([full_name, username, email, password, confirm_password]):
            flash('Please fill all required fields', 'error')
            return redirect(url_for('register_user'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register_user'))
        
        if not validate_password(password):
            flash('Password must contain only letters and numbers, minimum 6 characters', 'error')
            return redirect(url_for('register_user'))
        
        if not validate_gmail(email):
            flash('Please use a valid Gmail address', 'error')
            return redirect(url_for('register_user'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register_user'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register_user'))
        
        # Get phone number
        phone = request.form.get('phone')
        
        # Validate phone if provided
        if phone and not validate_phone(phone):
            flash('Please enter a valid phone number', 'error')
            return redirect(url_for('register_user'))

        # Create user
        user = User(
            full_name=full_name,
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            phone_number=phone
        )
        
        # Assign user role
        user_role = Role.query.filter_by(name='user').first()
        if user_role:
            user.roles.append(user_role)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register_user.html')

@app.route('/register/doctor', methods=['GET', 'POST'])
def register_doctor():
    """Doctor registration route"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        specialization = request.form.get('specialization')
        license_number = request.form.get('license_number')
        experience = request.form.get('experience')
        
        # Validation
        if not all([full_name, username, email, password, confirm_password, specialization, license_number]):
            flash('Please fill all required fields', 'error')
            return redirect(url_for('register_doctor'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register_doctor'))
        
        if not validate_password(password):
            flash('Password must contain only letters and numbers, minimum 6 characters', 'error')
            return redirect(url_for('register_doctor'))
        
        if not validate_gmail(email):
            flash('Please use a valid Gmail address', 'error')
            return redirect(url_for('register_doctor'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register_doctor'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register_doctor'))
        
        # Get phone number
        phone = request.form.get('phone')
        
        # Validate phone if provided
        if phone and not validate_phone(phone):
            flash('Please enter a valid phone number', 'error')
            return redirect(url_for('register_doctor'))

        # Create doctor
        user = User(
            full_name=full_name,
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            specialization=specialization,
            license_number=license_number,
            phone_number=phone,
            experience=experience
        )
        
        # Assign doctor role
        doctor_role = Role.query.filter_by(name='doctor').first()
        if doctor_role:
            user.roles.append(doctor_role)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Doctor registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register_doctor.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout route"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

# Dashboard Routes
@app.route('/dashboard')
@login_required
def dashboard():
    """Redirect to appropriate dashboard based on user role"""
    if current_user.has_role('admin'):
        return redirect(url_for('admin_dashboard'))
    elif current_user.has_role('doctor'):
        return redirect(url_for('doctor_dashboard'))
    else:
        return redirect(url_for('user_dashboard'))

@app.route('/user/dashboard')
@login_required
@role_required('user')
def user_dashboard():
    """User dashboard route"""
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    # Calculate normal and abnormal predictions
    normal_predictions = Prediction.query.filter_by(user_id=current_user.id).filter(Prediction.prediction_result.contains('[ Normal ]')).count()
    abnormal_predictions = total_predictions - normal_predictions
    # total_predictions = Prediction.query.count()
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(5).all()
    upcoming_appointments = Appointment.query.filter_by(user_id=current_user.id).filter(Appointment.appointment_datetime >= datetime.now()).count()
    total_doctors = User.query.filter(User.roles.any(Role.name == 'doctor')).count()
    total_users = User.query.filter(User.roles.any(Role.name == 'user')).count()
    total_contact_messages = ContactMessage.query.count()
    # appointments = Appointment.query.filter_by(user_id=current_user.id).order_by(Appointment.appointment_datetime.desc()).all()

    appointments = Appointment.query.filter_by(user_id=current_user.id).order_by(Appointment.created_at.desc()).limit(20).all()
    total_appointments = Appointment.query.filter_by(user_id=current_user.id).count()

    return render_template('user_dashboard.html',
                         total_predictions=total_predictions,
                         normal_predictions=normal_predictions,
                         abnormal_predictions=abnormal_predictions,
                         recent_predictions=recent_predictions,
                         upcoming_appointments=upcoming_appointments,
                         total_doctors=total_doctors,
                         total_users=total_users,
                         total_contact_messages=total_contact_messages,
                         appointments=appointments,
                         total_appointments=total_appointments)

@app.route('/doctor/dashboard')
@login_required
@role_required('doctor')
def doctor_dashboard():
    """Doctor dashboard route"""
    total_patients = db.session.query(db.func.count(db.distinct(Appointment.user_id))).filter(Appointment.doctor_id == current_user.id).scalar()

    # appointments = Appointment.query.filter_by(doctor_id=current_user.id).order_by(Appointment.appointment_datetime.desc()).all()
    appointments = Appointment.query.filter_by(doctor_id=current_user.id).order_by(Appointment.created_at.desc()).all()

    upcoming_appointments = Appointment.query.filter_by(doctor_id=current_user.id)\
        .filter(Appointment.appointment_datetime >= datetime.now())\
        .order_by(Appointment.appointment_datetime.asc()).all()

    past_appointments = Appointment.query.filter_by(doctor_id=current_user.id)\
        .filter(Appointment.appointment_datetime < datetime.now())\
        .order_by(Appointment.appointment_datetime.desc()).limit(10).all()

    recent_reports = Prediction.query.join(Appointment).filter(Appointment.doctor_id == current_user.id).order_by(Prediction.created_at.desc()).limit(5).all()

    # Calculate statistics for doctor dashboard
    total_predictions = Prediction.query.join(Appointment).filter(Appointment.doctor_id == current_user.id).count()
    normal_predictions = Prediction.query.join(Appointment).filter(Appointment.doctor_id == current_user.id).filter(Prediction.prediction_result.contains('[ Normal ]')).count()
    abnormal_predictions = total_predictions - normal_predictions
    total_appointments = Appointment.query.filter_by(doctor_id=current_user.id).count()

    recent_predictions = Prediction.query.join(Appointment).filter(Appointment.doctor_id == current_user.id).order_by(Prediction.created_at.desc()).limit(10).all()

    # Get additional statistics
    total_doctors = User.query.filter(User.roles.any(Role.name == 'doctor')).count()
    total_users = User.query.filter(User.roles.any(Role.name == 'user')).count()
    total_contact_messages = ContactMessage.query.count()

    return render_template('doctor_dashboard.html',
                         total_patients=total_patients,
                         appointments=appointments,
                         upcoming_appointments=upcoming_appointments,
                         past_appointments=past_appointments,
                         recent_reports=recent_reports,
                         total_predictions=total_predictions,
                         normal_predictions=normal_predictions,
                         abnormal_predictions=abnormal_predictions,
                         total_appointments=total_appointments,
                         recent_predictions=recent_predictions,
                         total_doctors=total_doctors,
                         total_users=total_users,
                         total_contact_messages=total_contact_messages)

@app.route('/admin/dashboard')
@login_required
@role_required('admin')
def admin_dashboard():
    """Admin dashboard route"""
    total_users = User.query.filter(User.roles.any(Role.name == 'user')).count()
    total_doctors = User.query.filter(User.roles.any(Role.name == 'doctor')).count()
    total_predictions = Prediction.query.count()
    # Calculate normal and abnormal predictions
    normal_predictions = Prediction.query.filter(Prediction.prediction_result.contains('[ Normal ]')).count()
    abnormal_predictions = total_predictions - normal_predictions
    total_appointments = Appointment.query.count()
    total_contact_messages = ContactMessage.query.count()
    recent_messages = ContactMessage.query.order_by(ContactMessage.created_at.desc()).limit(5).all()
    
    # Get prediction data for charts
    predictions_by_disease = db.session.query(
        Prediction.disease_type,
        db.func.count(Prediction.id)
    ).group_by(Prediction.disease_type).all()
    
    prediction_data = [[disease_type, count] for disease_type, count in predictions_by_disease]
    
    # Get users and doctors for management with counts
    users = User.query.filter(User.roles.any(Role.name == 'user')).all()
    
    # Calculate prediction and appointment counts for each user
    user_counts = {}
    for user in users:
        prediction_count = Prediction.query.filter_by(user_id=user.id).count()
        appointment_count = Appointment.query.filter_by(user_id=user.id).count()
        user_counts[user.id] = {
            'prediction_count': prediction_count,
            'appointment_count': appointment_count
        }
    
    doctors = User.query.filter(User.roles.any(Role.name == 'doctor')).all()
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(20).all()
    appointments = Appointment.query.order_by(Appointment.created_at.desc()).limit(20).all()
    recent_contacts = ContactMessage.query.order_by(ContactMessage.created_at.desc()).limit(10).all()
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         total_doctors=total_doctors,
                         total_predictions=total_predictions,
                         normal_predictions=normal_predictions,
                         abnormal_predictions=abnormal_predictions,
                         total_appointments=total_appointments,
                         total_contact_messages=total_contact_messages,
                         recent_messages=recent_messages,
                         prediction_data=prediction_data,
                         users=users,
                         user_counts=user_counts,
                         doctors=doctors,
                         recent_predictions=recent_predictions,
                         appointments=appointments,
                         recent_contacts=recent_contacts)

# Profile Routes
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management"""
    if request.method == 'POST':
        current_user.full_name = request.form.get('full_name')
        current_user.email = request.form.get('email')
        
        if current_user.has_role('doctor'):
            current_user.specialization = request.form.get('specialization')
            current_user.license_number = request.form.get('license_number')
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html')

# Disease Prediction Routes
@app.route('/ecg_predict', methods=['GET', 'POST'])
@login_required
@role_required('user')
def ecg_predict():
    """ECG prediction route"""
    if request.method == 'POST':
        # Get form data
        patient_name = request.form.get('patient_name')
        patient_age = request.form.get('patient_age')
        patient_gender = request.form.get('patient_gender')
        patient_id = request.form.get('patient_id')
        doctor_name = request.form.get('doctor_name')
        doctor_type = request.form.get('doctor_type')
        patient_symptoms = request.form.get('patient_symptoms')
        disease_type = "ECG"
        file = request.files.get('file')
        
        # Store form data in session for use in handle_ecg_prediction
        session['ecg_form_data'] = {
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'patient_id': patient_id,
            'doctor_name': doctor_name,
            'doctor_type': doctor_type,
            'patient_symptoms': patient_symptoms,
            'disease_type': disease_type,
            'file': file
        }
        
        report = handle_ecg_prediction(current_user.id)
        # Store ECG results in session to avoid URL parameter conversion issues
        session['ecg_report'] = report
        # Remove ecg_form_data from session after use
        session.pop('ecg_form_data', None)
        return redirect(url_for('ecg_result'))
    return render_template('ecg_predict.html')
        
@app.route('/ecg_result', methods=['GET'])
@login_required
def ecg_result():
    """ECG result route"""
    # Retrieve ECG results from session
    report = session.get('ecg_report')
    
    if not report:
        flash('No ECG results found. Please upload an ECG image first.', 'error')
        return redirect(url_for('ecg_predict'))
    
    # Clear the session data after retrieval
    session.pop('ecg_report', None)
    
    return render_template('ecg_result.html',
                           original=report['original'],
                           pred1=report['pred1'],
                           pred2=report['pred2'],
                           patient_name=report['patient_name'],
                           patient_age=report['patient_age'],
                           patient_gender=report['patient_gender'],
                           patient_id=report['patient_id'],
                           doctor_name=report['doctor_name'],
                           doctor_type=report['doctor_type'],
                           disease_type=report['disease_type'],
                           patient_symptoms=report['patient_symptoms'],
                           predicted_classes1=report['predicted_classes1'],
                           predicted_classes2=report['predicted_classes2'],
                           ecg_prediction=report['ecg_prediction'],
                           ecg_outputs=report['ecg_outputs'],
                           ecg_1d_signal=report['ecg_1d_signal'],
                           ecg_reduced=report['ecg_reduced'])



@app.route('/braintumor_predict', methods=['GET', 'POST'])
@login_required
@role_required('user')
def braintumor_predict():
    """Brain tumor prediction route"""
    if request.method == 'POST':
        # Get form data
        patient_name = request.form.get('patient_name')
        patient_age = request.form.get('patient_age')
        patient_gender = request.form.get('patient_gender')
        patient_id = request.form.get('patient_id')
        doctor_name = request.form.get('doctor_name')
        doctor_type = request.form.get('doctor_type')
        patient_symptoms = request.form.get('patient_symptoms')
        disease_type = "Brain Tumor"
        file = request.files.get('file')
        
        # Store form data in session for use in handle_braintumor_prediction
        session['braintumor_form_data'] = {
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'patient_id': patient_id,
            'doctor_name': doctor_name,
            'doctor_type': doctor_type,
            'patient_symptoms': patient_symptoms,
            'disease_type': disease_type,
            'file': file
        }

        print(session['braintumor_form_data'])
        
        report = handle_braintumor_prediction(current_user.id)
        # Store brain tumor results in session to avoid URL parameter conversion issues
        session['braintumor_report'] = report
        # Remove braintumor_form_data from session after use
        session.pop('braintumor_form_data', None)
        return redirect(url_for('braintumor_result'))
    return render_template('braintumor_predict.html')

@app.route('/braintumor_result', methods=['GET'])
@login_required
def braintumor_result():
    """brain tumor result route"""
    # Retrieve ECG results from session
    report = session.get('braintumor_report')
    
    if not report:
        flash('No Brain Tumor results found. Please upload an Brain Tumor image first.', 'error')
        return redirect(url_for('braintumor_predict'))
    
    # Clear the session data after retrieval
    session.pop('braintumor_report', None)
    
    return render_template('braintumor_result.html',
                           original_image=report['original_image'],
                           predicted_image=report['predicted_image'],
                           patient_name=report['patient_name'],
                           patient_age=report['patient_age'],
                           patient_gender=report['patient_gender'],
                           patient_id=report['patient_id'],
                           doctor_name=report['doctor_name'],
                           doctor_type=report['doctor_type'],
                           disease_type=report['disease_type'],
                           patient_symptoms=report['patient_symptoms'],
                           prediction_result=report['prediction_result'])




@app.route('/pneumonia_predict', methods=['GET', 'POST'])
@login_required
@role_required('user')
def pneumonia_predict():
    """Pneumonia prediction route"""
    if request.method == 'POST':
        # Get form data
        patient_name = request.form.get('patient_name')
        patient_age = request.form.get('patient_age')
        patient_gender = request.form.get('patient_gender')
        patient_id = request.form.get('patient_id')
        doctor_name = request.form.get('doctor_name')
        doctor_type = request.form.get('doctor_type')
        patient_symptoms = request.form.get('patient_symptoms')
        disease_type = "Pneumonia"
        file = request.files.get('file')
        
        # Store form data in session for use in handle_braintumor_prediction
        session['pneumonia_form_data'] = {
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'patient_id': patient_id,
            'doctor_name': doctor_name,
            'doctor_type': doctor_type,
            'patient_symptoms': patient_symptoms,
            'disease_type': disease_type,
            'file': file
        }

        print(session['pneumonia_form_data'])
        
        report = handle_pneumonia_prediction(current_user.id)
        # Store pneumonia results in session to avoid URL parameter conversion issues
        session['pneumonia_report'] = report
        # Remove pneumonia_form_data from session after use
        session.pop('pneumonia_form_data', None)
        return redirect(url_for('pneumonia_result'))
    return render_template('pneumonia_predict.html')

@app.route('/pneumonia_result', methods=['GET'])
@login_required
def pneumonia_result():
    """pneumonia result route"""
    # Retrieve ECG results from session
    report = session.get('pneumonia_report')
    
    if not report:
        flash('No Pneumonia results found. Please upload an Pneumonia image first.', 'error')
        return redirect(url_for('pneumonia_predict'))
    
    # Clear the session data after retrieval
    session.pop('pneumonia_report', None)
    
    return render_template('pneumonia_result.html',
                           original_image=report['original_image'],
                           predicted_image=report['predicted_image'],
                           patient_name=report['patient_name'],
                           patient_age=report['patient_age'],
                           patient_gender=report['patient_gender'],
                           patient_id=report['patient_id'],
                           doctor_name=report['doctor_name'],
                           doctor_type=report['doctor_type'],
                           disease_type=report['disease_type'],
                           patient_symptoms=report['patient_symptoms'],
                           prediction_result=report['prediction_result'])





@app.route('/lungcancer_predict', methods=['GET', 'POST'])
@login_required
@role_required('user')
def lungcancer_predict():
    """Lung cancer prediction route"""
    if request.method == 'POST':
        # Get form data
        patient_name = request.form.get('patient_name')
        patient_age = request.form.get('patient_age')
        patient_gender = request.form.get('patient_gender')
        patient_id = request.form.get('patient_id')
        doctor_name = request.form.get('doctor_name')
        doctor_type = request.form.get('doctor_type')
        patient_symptoms = request.form.get('patient_symptoms')
        disease_type = "Lung Cancer"
        file = request.files.get('file')
        
        # Store form data in session for use in handle_lungcancer_prediction
        session['lungcancer_form_data'] = {
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'patient_id': patient_id,
            'doctor_name': doctor_name,
            'doctor_type': doctor_type,
            'patient_symptoms': patient_symptoms,
            'disease_type': disease_type,
            'file': file
        }

        print(session['lungcancer_form_data'])
        
        report = handle_lungcancer_prediction(current_user.id)
        # Store lungcancer results in session to avoid URL parameter conversion issues
        session['lungcancer_report'] = report
        # Remove lungcancer_form_data from session after use
        session.pop('lungcancer_form_data', None)
        return redirect(url_for('lungcancer_result'))
    return render_template('lungcancer_predict.html')

@app.route('/lungcancer_result', methods=['GET'])
@login_required
def lungcancer_result():
    """lungcancer result route"""
    # Retrieve ECG results from session
    report = session.get('lungcancer_report')
    
    if not report:
        flash('No Lung Cancer results found. Please upload an Lung Cancer image first.', 'error')
        return redirect(url_for('lungcancer_predict'))
    
    # Clear the session data after retrieval
    session.pop('lungcancer_report', None)
    
    return render_template('lungcancer_result.html',
                           original_image=report['original_image'],
                           predicted_image=report['predicted_image'],
                           patient_name=report['patient_name'],
                           patient_age=report['patient_age'],
                           patient_gender=report['patient_gender'],
                           patient_id=report['patient_id'],
                           doctor_name=report['doctor_name'],
                           doctor_type=report['doctor_type'],
                           disease_type=report['disease_type'],
                           patient_symptoms=report['patient_symptoms'],
                           prediction_result=report['prediction_result'])




# Appointment Routes
@app.route('/user/appointments')
@login_required
@role_required('user')
def user_appointments():
    """Display user's appointments"""
    appointments = Appointment.query.filter_by(user_id=current_user.id).order_by(Appointment.appointment_datetime.desc()).all()
    return render_template('user_appointments.html', appointments=appointments)

@app.route('/book/appointment', methods=['GET', 'POST'])
@login_required
@role_required('user')
def book_appointment():
    """Book a new appointment"""
    if request.method == 'POST':
        try:
            # Get form data
            doctor_id = request.form.get('doctor_id')
            prediction_id = request.form.get('prediction_id')
            appointment_date = request.form.get('appointment_date')
            appointment_time = request.form.get('appointment_time')
            notes = request.form.get('notes')
            
            # Validate required fields
            if not all([doctor_id, appointment_date, appointment_time]):
                flash('Please fill all required fields', 'error')
                return redirect(url_for('book_appointment'))
            
            # Get doctor
            doctor = User.query.get(doctor_id)
            if not doctor or not doctor.has_role('doctor'):
                flash('Invalid doctor selected', 'error')
                return redirect(url_for('book_appointment'))
            
            # Parse date and time
            appointment_datetime = datetime.strptime(f"{appointment_date} {appointment_time}", "%Y-%m-%d %H:%M")
            
            # Validate appointment date is not in the past
            if appointment_datetime < datetime.now():
                flash('Appointment date and time cannot be in the past', 'error')
                return redirect(url_for('book_appointment'))
            
            # Validate appointment date is after prediction date if prediction is linked
            if prediction_id:
                prediction = Prediction.query.get(prediction_id)
                if prediction and prediction.user_id == current_user.id:
                    if appointment_datetime <= prediction.created_at:
                        flash('Appointment date and time must be after the prediction date', 'error')
                        return redirect(url_for('book_appointment'))
            
            # Create appointment
            appointment = Appointment(
                user_id=current_user.id,
                doctor_id=doctor_id,
                prediction_id=prediction_id if prediction_id else None,
                appointment_date=appointment_datetime.date(),
                appointment_time=appointment_datetime.time(),
                appointment_datetime=appointment_datetime,
                notes=notes,
                status='pending'
            )
            
            # If prediction is linked, copy patient details
            if prediction_id:
                prediction = Prediction.query.get(prediction_id)
                if prediction and prediction.user_id == current_user.id:
                    appointment.patient_name = prediction.patient_name
                    appointment.patient_age = prediction.patient_age
                    appointment.patient_gender = prediction.patient_gender
                    appointment.referring_doctor = prediction.referring_doctor
                    appointment.symptoms = prediction.symptoms
                    appointment.disease_type = prediction.disease_type
                    appointment.prediction_result = prediction.prediction_result
                    appointment.original_image = prediction.original_image
                    appointment.predicted_image = prediction.predicted_image
            
            # Set doctor details
            appointment.doctor_name = doctor.full_name
            appointment.specialization = doctor.specialization or 'General'
            appointment.license_number = doctor.license_number
            appointment.doctor_phone = doctor.phone_number
            appointment.experience = doctor.experience
            
            # If no prediction linked, use current user details
            if not prediction_id:
                appointment.patient_name = current_user.full_name
                appointment.referring_doctor = 'Self-referred'
            
            db.session.add(appointment)
            db.session.commit()
            
            flash('Appointment booked successfully!', 'success')
            return redirect(url_for('user_appointments'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error booking appointment: {str(e)}', 'error')
            return redirect(url_for('book_appointment'))
    
    # GET request - show booking form
    doctors = User.query.filter(User.roles.any(Role.name == 'doctor')).all()
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    appointments = Appointment.query.filter_by(user_id=current_user.id).order_by(Appointment.appointment_datetime.desc()).all()

    return render_template('book_appointment.html',
                         doctors=doctors,
                         predictions=predictions,
                         appointments=appointments)

@app.route('/prediction/<int:prediction_id>')
@login_required
def prediction_result(prediction_id):
    """Display prediction result based on prediction ID"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Check if the current user is authorized to view this prediction
    if current_user.has_role('user') and prediction.user_id != current_user.id:
        flash('Access denied. You are not authorized to view this prediction.', 'error')
        return redirect(url_for('user_dashboard'))
    

    original_image_path = prediction.original_image.split("\\")[-1] if prediction.original_image else ''
    predicted_image_path = prediction.predicted_image.split("\\")[-1] if prediction.predicted_image else ''
    
    prediction_data = {
        'patient_name': prediction.patient_name,
        'patient_age': prediction.patient_age,
        'patient_gender': prediction.patient_gender,
        'patient_id': prediction.patient_id,

        'doctor_name': prediction.referring_doctor,
        'disease_type': prediction.disease_type,
        'patient_symptoms': prediction.symptoms,

        'prediction_result': prediction.prediction_result,
        'original_image': original_image_path,
        'predicted_image': predicted_image_path
    }
    
    return render_template('prediction.html', **prediction_data)


@app.route('/original/<int:original_id>')
@login_required
def original_image(original_id):
    """Display image based on image path"""

    # If you're passing full path
    prediction = Prediction.query.get_or_404(original_id)
    original_image_path = prediction.original_image.split("\\")[-1] if prediction.original_image else ''

    viewimage_data = {
        'image': original_image_path
    }
    
    return render_template('original_image.html', **viewimage_data)


@app.route('/predicted/<int:predicted_id>')
@login_required
def predicted_image(predicted_id):
    """Display image based on image path"""

    # If you're passing full path
    prediction = Prediction.query.get_or_404(predicted_id)
    predicted_image_path = prediction.predicted_image.split("\\")[-1] if prediction.predicted_image else ''

    viewimage_data = {
        'image': predicted_image_path
    }
    
    return render_template('predicted_image.html', **viewimage_data)






@app.route('/user_report_data')
@login_required
def user_report_data():
    """User report data page"""
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    # Calculate normal and abnormal predictions
    normal_predictions = Prediction.query.filter_by(user_id=current_user.id).filter(Prediction.prediction_result.contains('[ Normal ]')).count()
    abnormal_predictions = total_predictions - normal_predictions
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(5).all()
    upcoming_appointments = Appointment.query.filter_by(user_id=current_user.id).filter(Appointment.appointment_datetime >= datetime.now()).count()
    total_doctors = User.query.filter(User.roles.any(Role.name == 'doctor')).count()
    appointments = Appointment.query.filter_by(user_id=current_user.id).order_by(Appointment.appointment_datetime.desc()).all()

    return render_template('user_report_data.html',
                         total_predictions=total_predictions,
                         normal_predictions=normal_predictions,
                         abnormal_predictions=abnormal_predictions,
                         recent_predictions=recent_predictions,
                         upcoming_appointments=upcoming_appointments,
                         total_doctors=total_doctors,
                         appointments=appointments)

# Database initialization
def init_db():
    """Initialize database with roles and admin user"""
    with app.app_context():
        db.create_all()
        
        # Create roles
        roles = ['admin', 'doctor', 'user']
        for role_name in roles:
            if not Role.query.filter_by(name=role_name).first():
                role = Role(name=role_name, description=f'{role_name.capitalize()} role')
                db.session.add(role)
        
        # Create admin user
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin_role = Role.query.filter_by(name='admin').first()
            admin = User(
                full_name='Admin User',
                username='admin',
                email='admin@cliniq.ai',
                password_hash=generate_password_hash('admin123'),
                active=True
            )
            admin.roles.append(admin_role)
            db.session.add(admin)
        
        db.session.commit()
        print("Database initialized successfully")

# Initialize ML models on startup
# load_ml_models()

if __name__ == '__main__':
    init_db()
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
