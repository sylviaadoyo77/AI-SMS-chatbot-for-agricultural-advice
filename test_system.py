import requests
import json

url = "http://localhost:5000/api/process-query"
payload = {"query": "How do I treat aphids on my maize plants?"}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Make sure Flask is running on port 5000.")
except Exception as e:
    print(f"Error: {e}")