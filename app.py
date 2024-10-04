import os
import logging
import numpy as np
import cv2  # OpenCV for video frames
import librosa  # For audio processing
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='frontend', static_url_path='')

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

limiter = Limiter(key_func=get_remote_address, app=app)

# Load models once
image_model = tf.keras.models.load_model('deepfake_image_detection_model.h5')
audio_model = tf.keras.models.load_model('deepfake_audio_detection_model.h5')

@app.route('/')
def home():
    return send_from_directory('frontend', 'index.html')

# Function to load and preprocess images
def load_images(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not load image from {file_path}")
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)  # Shape (1, 224, 224, 3)

# Function to load and preprocess video frames
def load_video_frames(file_path, frame_count=100):
    cap = cv2.VideoCapture(file_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // frame_count, 1)

    for _ in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(_ * frame_interval, total_frames - 1))
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype('float32') / 255.0
        frames.append(frame)

    cap.release()
    if frames:
        return np.array(frames)  # Shape (num_frames, 224, 224, 3)
    else:
        raise ValueError(f"No frames extracted from video {file_path}")

# Function to load and preprocess audio
def load_audio(file_path):
    SAMPLE_RATE = 16000
    MAX_DURATION = 5 * SAMPLE_RATE
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Pad or truncate to fixed length (5 seconds)
    if len(signal) < MAX_DURATION:
        signal = np.pad(signal, (0, MAX_DURATION - len(signal)), "constant")
    else:
        signal = signal[:MAX_DURATION]

    return np.expand_dims(signal, axis=-1)[None, :, :]  # Shape (1, 80000, 1)

@app.route('/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded or file name is empty'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(file_path)
        logger.info(f"File saved to {file_path}")

        # Process file based on its type
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Process image
            image_data = load_images(file_path)
            prediction = image_model.predict(image_data)[0][0]
            real_percentage = prediction * 100
            fake_percentage = (1 - prediction) * 100

        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Process video (using sampled frames)
            video_data = load_video_frames(file_path, frame_count=20)  # Sample only 20 frames
            predictions = image_model.predict(video_data)  # Predict on all frames at once
            avg_prediction = np.mean(predictions)

            real_percentage = avg_prediction * 100
            fake_percentage = (1 - avg_prediction) * 100

        elif filename.lower().endswith(('.mp3', '.wav', '.flac')):
            # Process audio
            audio_data = load_audio(file_path)
            prediction = audio_model.predict(audio_data)[0][0]
            real_percentage = prediction * 100
            fake_percentage = (1 - prediction) * 100

        # Log percentages in the console
        logger.info(f"Real Percentage: {round(real_percentage, 2)}%")
        logger.info(f"Fake Percentage: {round(fake_percentage, 2)}%")

        result = {
            'real_percentage': round(real_percentage, 2),
            'fake_percentage': round(fake_percentage, 2),
            'message': f'The file is {"fake" if fake_percentage > 50 else "real"}'
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error("Error processing file:", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
