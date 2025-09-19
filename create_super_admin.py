# create_super_admin.py
import mysql.connector
import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash

# Load environment variables
load_dotenv()

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

def create_super_admin():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if super admin already exists
    cursor.execute("SELECT id FROM admin_users WHERE role = 'super_admin'")
    if cursor.fetchone():
        print("Super admin already exists.")
        return
    
    # Create super admin
    admin_id = f"ADM_SUPER_{os.urandom(4).hex()}"
    username = "superadmin"
    password = "superadmin123"  # Change this in production!
    hashed_password = generate_password_hash(password)
    
    cursor.execute(
        "INSERT INTO admin_users (id, username, full_name, email, password_hash, role, is_active, registration_status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (admin_id, username, "Super Administrator", "super@agrichatbot.com", hashed_password, "super_admin", 1, 'approved')
    )
    
    conn.commit()
    print("Super admin created successfully!")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print("Please change the password after first login!")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_super_admin()