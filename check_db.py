import sqlite3
import os

def check_predictions():
    # Check if database file exists
    db_path = 'instance/cliniq_ai.db'
    print(f"Database file exists: {os.path.exists(db_path)}")
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check predictions and their user_ids
            cursor.execute("SELECT * FROM prediction")
            predictions = cursor.fetchall()
            print("\nPredictions in database:")
            for pred in predictions:
                print(pred)
                
            # Check users
            cursor.execute("SELECT * FROM user")
            users = cursor.fetchall()
            print("\nUsers in database:")
            for user in users:
                print(user)

            # Check appointments
            cursor.execute("SELECT * FROM appointment")
            users = cursor.fetchall()
            print("\nAppointment in database:")
            for user in users:
                print(user)
                
            conn.close()
        except Exception as e:
            print(f"Error accessing database: {e}")
    else:
        print("Database file not found")

if __name__ == "__main__":
    check_predictions()
