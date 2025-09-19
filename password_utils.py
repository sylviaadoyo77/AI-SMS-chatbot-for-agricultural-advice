#password.utils.py
from werkzeug.security import generate_password_hash
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python password_utils.py <password>")
        sys.exit(1)
    
    password = sys.argv[1]
    hashed = generate_password_hash(password)
    print(f"Password: {password}")
    print(f"Hashed: {hashed}")