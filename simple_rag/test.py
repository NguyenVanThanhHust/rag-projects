import requests
import os

# Assuming your FastAPI server is running on http://127.0.0.1:8000
BASE_URL = "http://127.0.0.1:8000"

def upload_single_file(file_path):
    """
    Uploads a single file to the /uploadfile/ endpoint.
    """
    url = f"{BASE_URL}/uploadfile/"
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        # The 'application/octet-stream' is a generic binary type.
        # requests can often infer more specific types if you don't provide it.
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            print(f"Single file upload response: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"Error uploading single file: {e}")

if __name__ == "__main__":
    test_file = "scenario1_cam4_short.txt"
    print("--- Testing Single File Upload ---")
    upload_single_file(test_file)
    print("\n")

    import requests

    # Define the API endpoint
    url = "http://localhost:8000/query/"

    # Define the query parameters
    params = {
        "video_name_prefix": "scenario1_cam4_short",
        "query": "What is the topic of the video?"
    }

    # Send the GET request
    response = requests.get(url, params=params)

    # Print the response (assuming it's JSON)
    print(response.json())