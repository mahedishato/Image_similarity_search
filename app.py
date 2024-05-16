import numpy as np
import cv2
import tensorflow as tf
from keras.applications import ResNet101
from keras.applications.resnet import preprocess_input
from keras.models import Model
from scipy.spatial.distance import cosine
from io import BytesIO
from PIL import Image
import base64
import io
from flask import Flask, Blueprint, jsonify, request
from flask import render_template

app = Flask(__name__)

def preprocess_image(image_path):
    # Preprocess the image for ResNet101 model
    binary_image= Image.open(image_path).convert("L")
    numpy_image = np.array(binary_image)
    rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2RGB)
    rgb_image = cv2.resize(rgb_image, (224, 224))
    img_array = np.array(rgb_image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def cosine_similarity(feature_vector1, feature_vector2):
    # Calculate cosine similarity between two feature vectors
    return 1 - cosine(feature_vector1, feature_vector2)

def compare_signatures(image_path1, image_path2):
    # Load pre-trained ResNet101 model
    base_model = ResNet101(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    
    # Preprocess images
    binary_image1 = preprocess_image(image_path1)
    binary_image2 = preprocess_image(image_path2)
    
    # Extract features
    feature_vector1 = model.predict(binary_image1).flatten()
    feature_vector2 = model.predict(binary_image2).flatten()
    
    # Calculate similarity score
    similarity_score = cosine_similarity(feature_vector1, feature_vector2)
    return similarity_score

def compare_signatures_base64(image_data1, image_data2):
    # Load pre-trained ResNet101 model
    base_model = ResNet101(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    
    # Decode base64 images
    image_stream1 = io.BytesIO(base64.b64decode(image_data1))
    image_stream2 = io.BytesIO(base64.b64decode(image_data2))
    
    # Preprocess images
    binary_image1 = preprocess_image(image_stream1)
    binary_image2 = preprocess_image(image_stream2)
    
    # Extract features
    feature_vector1 = model.predict(binary_image1).flatten()
    feature_vector2 = model.predict(binary_image2).flatten()
    
    # Calculate similarity score
    similarity_score = cosine_similarity(feature_vector1, feature_vector2)
    return similarity_score


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare_signatures', methods=['POST'])
def compare_signatures_endpoint():
    try:
        if 'signature1' not in request.files or 'signature2' not in request.files:
            return jsonify({"error": "Both 'signature1' and 'signature2' files are required."}), 400
        threshold = 0.85
        similarity = compare_signatures(request.files['signature1'], request.files['signature2'])
        if similarity >= threshold:
            response = {"match": True, "result": "same_signature"}
        else:
            response = {"match": False, "result": "different_signature"}
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# @app.route('/compare_signatures_base64', methods=['POST'])
# def compare_signatures_endpoint_base64():
#     try:
#         data = request.get_json()
#         if 'signature1' not in data or 'signature2' not in data:
#             return jsonify({"error": "Both 'signature1' and 'signature2' are required in base64 format."}), 400

#         similarity = compare_signatures_base64(data['signature1'], data['signature2'])

#         similarity_threshold_sig = 0.5  
#         response = {"match": True, "result": "same_signature"}
#         return jsonify(response), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")