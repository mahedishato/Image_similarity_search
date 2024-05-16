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

def compare_signature(image_path1, image_path2):
    print("image type: ", image_path1)
    # Load pre-trained ResNet101 model
    base_model = ResNet101(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    print("1")
    
    def preprocess_image(image_path):
        print("Image path:", image_path)
        print(type(image_path))
        binary_image= Image.open(image_path).convert("L")
        print("binary image:", binary_image)
        numpy_image = np.array(binary_image)
        rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2RGB)
        rgb_image = cv2.resize(rgb_image, (224, 224))
        img_array = np.array(rgb_image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        print("Preprocessed image shape:", img_array.shape)
        return img_array

    
    # Calculate cosine similarity
    def cosine_similarity(feature_vector1, feature_vector2):
        return 1 - cosine(feature_vector1, feature_vector2)
    
    # Main function logic
    def find_similarity(binary_image_path1, binary_image_path2):
        binary_image1 = preprocess_image(binary_image_path1)
        binary_image2 = preprocess_image(binary_image_path2)
        feature_vector1 = model.predict(binary_image1).flatten()
        feature_vector2 = model.predict(binary_image2).flatten()
        similarity_score = cosine_similarity(feature_vector1, feature_vector2)
        print("3")
        return similarity_score
    
    # Call main function
    similarity_score = find_similarity(image_path1, image_path2)
    
    return similarity_score


def compare_signatures_base64(image_path1, image_path2):
    print("image type: ", image_path1)
    # Load pre-trained ResNet101 model
    base_model = ResNet101(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    print("1")
    
    def preprocess_image(image_path):
        print("Image path:", image_path)
        print(type(image_path))
        image_data = base64.b64decode(image_path)
        image_stream = io.BytesIO(image_data)
        binary_image= Image.open(image_stream).convert("L")
        print("binary image:", binary_image)
        numpy_image = np.array(binary_image)
        rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2RGB)
        rgb_image = cv2.resize(rgb_image, (224, 224))
        img_array = np.array(rgb_image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        print("Preprocessed image shape:", img_array.shape)
        return img_array

    
    # Calculate cosine similarity
    def cosine_similarity(feature_vector1, feature_vector2):
        return 1 - cosine(feature_vector1, feature_vector2)
    
    # Main function logic
    def find_similarity(binary_image_path1, binary_image_path2):
        binary_image1 = preprocess_image(binary_image_path1)
        binary_image2 = preprocess_image(binary_image_path2)
        feature_vector1 = model.predict(binary_image1).flatten()
        feature_vector2 = model.predict(binary_image2).flatten()
        similarity_score = cosine_similarity(feature_vector1, feature_vector2)
        print("3")
        return similarity_score
    
    # Call main function
    similarity_score = find_similarity(image_path1, image_path2)
    
    return similarity_score
