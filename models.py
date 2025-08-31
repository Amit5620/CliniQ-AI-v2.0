from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_security import RoleMixin
from datetime import datetime

db = SQLAlchemy()

# User Roles association table
roles_users = db.Table('roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)

class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    active = db.Column(db.Boolean(), default=True)
    roles = db.relationship('Role', secondary=roles_users, backref=db.backref('users', lazy='dynamic'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional fields for doctors
    specialization = db.Column(db.String(100))
    license_number = db.Column(db.String(50))
    phone_number = db.Column(db.String(20))
    experience = db.Column(db.Integer, nullable=True)
    
    def has_role(self, role_name):
        return any(role.name == role_name for role in self.roles)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_age = db.Column(db.Integer, nullable=False)
    patient_gender = db.Column(db.String(10), nullable=False)
    patient_id = db.Column(db.String(50), nullable=False)
    referring_doctor = db.Column(db.String(100), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    disease_type = db.Column(db.String(50), nullable=False)  # ecg, brain_tumor, pneumonia, lung_cancer
    original_image = db.Column(db.String(200))
    predicted_image = db.Column(db.String(200))
    prediction_result = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='predictions')

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'), nullable=True)
    
    # Patient details
    patient_name = db.Column(db.String(100), nullable=False)
    patient_age = db.Column(db.Integer, nullable=False)
    patient_gender = db.Column(db.String(10), nullable=False)
    referring_doctor = db.Column(db.String(100), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    disease_type = db.Column(db.String(50), nullable=True)
    prediction_result = db.Column(db.String(100), nullable=True)
    original_image = db.Column(db.String(200), nullable=True)
    predicted_image = db.Column(db.String(200), nullable=True)
    
    # Doctor details
    doctor_name = db.Column(db.String(100), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    license_number = db.Column(db.String(50), nullable=True)
    doctor_phone = db.Column(db.String(20), nullable=True)
    experience = db.Column(db.Integer, nullable=True)
    
    # Appointment details
    appointment_date = db.Column(db.Date, nullable=False)
    appointment_time = db.Column(db.Time, nullable=False)
    appointment_datetime = db.Column(db.DateTime, nullable=False)  # Combined date and time
    status = db.Column(db.String(20), default='pending')  # pending, confirmed, completed, cancelled
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', foreign_keys=[user_id], backref='appointments')
    doctor = db.relationship('User', foreign_keys=[doctor_id], backref='doctor_appointments')
    prediction = db.relationship('Prediction', backref='appointment')

class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='unread')  # unread, read, replied
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
