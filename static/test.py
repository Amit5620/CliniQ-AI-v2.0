import json
import os
import sys
from werkzeug.security import generate_password_hash

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import User, Role

def load_sample_doctors():
    """Load sample doctors from JSON file and add them to the database."""

    with app.app_context():
        try:
            # Read the sample doctors JSON file
            with open('sample_doctors.json', 'r', encoding='utf-8') as file:
                doctors_data = json.load(file)

            # Get or create the doctor role
            doctor_role = Role.query.filter_by(name='doctor').first()
            if not doctor_role:
                doctor_role = Role(name='doctor', description='Medical Doctor')
                db.session.add(doctor_role)
                db.session.commit()
                print("✅ Created 'doctor' role")

            doctors_added = 0
            doctors_skipped = 0

            for doctor_data in doctors_data:
                # Check if doctor already exists by email or username
                existing_doctor = User.query.filter(
                    (User.email == doctor_data['email']) |
                    (User.username == doctor_data['username'])
                ).first()

                if existing_doctor:
                    print(f"⚠️  Doctor {doctor_data['full_name']} already exists (Email: {doctor_data['email']})")
                    doctors_skipped += 1
                    continue

                # Create new doctor user
                new_doctor = User(
                    full_name=doctor_data['full_name'],
                    username=doctor_data['username'],
                    email=doctor_data['email'],
                    password_hash=generate_password_hash(doctor_data['password']),
                    specialization=doctor_data['specialization'],
                    license_number=doctor_data['license_number'],
                    phone_number=doctor_data['phone_number'],
                    experience=int(doctor_data['experience'].split()[0]) if doctor_data['experience'] else None,
                    active=True
                )

                # Add doctor role to the user
                new_doctor.roles.append(doctor_role)

                # Add to database
                db.session.add(new_doctor)
                doctors_added += 1

                print(f"✅ Added doctor: {doctor_data['full_name']} ({doctor_data['specialization']})")

            # Commit all changes
            db.session.commit()

            print("\n🎉 Successfully completed!")
            print(f"📊 Doctors added: {doctors_added}")
            print(f"⏭️  Doctors skipped (already exist): {doctors_skipped}")
            print(f"📋 Total doctors in database: {User.query.filter(User.roles.any(name='doctor')).count()}")

        except FileNotFoundError:
            print("❌ Error: sample_doctors.json file not found!")
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON format in sample_doctors.json - {e}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            db.session.rollback()

if __name__ == "__main__":
    print("🏥 CliniQ AI - Sample Doctors Database Loader")
    print("=" * 50)
    load_sample_doctors()
