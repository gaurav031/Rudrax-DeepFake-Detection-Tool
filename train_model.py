import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Directories for real and fake images and audio
REAL_IMAGE_DIR = "data/real/videos/"
FAKE_IMAGE_DIR = "data/fake/videos/"
REAL_AUDIO_DIR = "data/real/audios/"
FAKE_AUDIO_DIR = "data/fake/audios/"

# Image and audio dimensions and other constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Check if directories exist
for directory in [REAL_IMAGE_DIR, FAKE_IMAGE_DIR, REAL_AUDIO_DIR, FAKE_AUDIO_DIR]:
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist.")

# Create an ImageDataGenerator for loading and preprocessing images
image_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    validation_split=0.2  # 20% data for validation
)

# Load training and validation data for images
train_image_generator = image_datagen.flow_from_directory(
    REAL_IMAGE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  
    subset='training',
    shuffle=True
)

validation_image_generator = image_datagen.flow_from_directory(
    FAKE_IMAGE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Function to load and preprocess audio files from a directory
def audio_file_generator(real_audio_dir, fake_audio_dir, batch_size):
    categories = [0, 1]  # 0 for real, 1 for fake
    while True:
        for label, category_dir in zip(categories, [real_audio_dir, fake_audio_dir]):
            for file in os.listdir(category_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(category_dir, file)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    # Pad or truncate to fixed length
                    if len(signal) < SAMPLES_PER_TRACK:
                        signal = np.pad(signal, (0, max(0, SAMPLES_PER_TRACK - len(signal))), "constant")
                    else:
                        signal = signal[:SAMPLES_PER_TRACK]

                    yield np.expand_dims(signal, axis=-1), label  # Return audio data and label

# Define the model architecture for images
def create_image_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train image model
image_model = create_image_model()
image_model.fit(train_image_generator, epochs=EPOCHS, validation_data=validation_image_generator)

# Save image model
image_model.save('deepfake_image_detection_model.h5')

# Create and train audio model
audio_model = tf.keras.Sequential([
    layers.Input(shape=(SAMPLES_PER_TRACK, 1)),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

audio_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create audio data generator
audio_generator = audio_file_generator(REAL_AUDIO_DIR, FAKE_AUDIO_DIR, BATCH_SIZE)

# Train audio model
audio_model.fit(audio_generator, steps_per_epoch=len(os.listdir(REAL_AUDIO_DIR)) // BATCH_SIZE + 1,
                validation_steps=len(os.listdir(FAKE_AUDIO_DIR)) // BATCH_SIZE + 1, epochs=EPOCHS)

# Save audio model
audio_model.save('deepfake_audio_detection_model.h5') 