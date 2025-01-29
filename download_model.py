import requests
file_id = "1N3hFvwwQ50THJ6MpyDeMASMOScm6V4W-"
download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
file_name = "shape_predictor_68_face_landmarks.dat"
print(f"Downloading {file_name}...")
response = requests.get(download_url, stream=True)

with open(file_name, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            file.write(chunk)

print(f"Download completed: {file_name}")
