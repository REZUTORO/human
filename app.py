from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'imageFile' not in request.files:
        return "No file part", 400
    file = request.files['imageFile']
    if file.filename == '':
        return "No selected file", 400

    # Read image in memory
    npimg = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Perform face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    _, buffer = cv2.imencode('.jpg', img)
    response_img = buffer.tobytes()

    return response_img, 200, {'Content-Type': 'image/jpeg'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
