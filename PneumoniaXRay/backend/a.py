import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model from the pickle file
with open('models/cnn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to preprocess a single image
def preprocess_image(image_path, img_size=150):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to expected input size
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Reshape to match model input shape (batch_size, height, width, channels)
    img_ready = img_normalized.reshape(1, img_size, img_size, 1)
    
    return img_ready

# Example usage
def predict_pneumonia(image_path):
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_img)[0][0]
    print(f"Raw prediction: {prediction}")
    
    # Interpret results (based on your model's output format)
    result = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return result, confidence

# Example call
result, confidence = predict_pneumonia("data/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg")
print(f"Diagnosis: {result} (Confidence: {confidence:.2f})")