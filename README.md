Certainly! Here's a README template for your GitHub repository with detailed documentation:

---

# Signature Similarity Detection using Deep Learning

This repository contains Python code for detecting similarity between signatures using deep learning techniques. It utilizes the ResNet101 architecture pre-trained on the ImageNet dataset to extract feature vectors from signature images and then computes the cosine similarity between these feature vectors to determine the similarity between signatures.

## Requirements

- Python 3.10.12
- numpy
- opencv-python (cv2)
- TensorFlow 2.x
- scipy

You can install the required dependencies via pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/mahedishato/Image_similarity_search.git
```

2. Prepare your signature images. Ensure that they are binary images in JPEG format.

3. Update the `image_path1` and `image_path2` variables in the `main.py` file with the paths to your signature images.

4. Set the threshold for similarity in the `threshold` variable.

5. Run the script:

```bash
python main.py
```

## Methodology

1. **Preprocessing**: The input signature images are preprocessed to convert them to RGB format and resize them to the input size required by the ResNet101 model (224x224 pixels).

2. **Feature Extraction**: The ResNet101 model is used as a feature extractor. The model is loaded with pre-trained weights from the ImageNet dataset, and the output of the average pooling layer is extracted as the feature vector for each signature image.

3. **Similarity Computation**: Cosine similarity is calculated between the feature vectors of the two signature images. Cosine similarity measures the cosine of the angle between two vectors and ranges from -1 to 1, where 1 indicates identical vectors and -1 indicates completely opposite vectors.

4. **Thresholding**: The similarity score is compared with a predefined threshold. If the similarity score is above the threshold, the signatures are considered a match; otherwise, they are considered unmatched.

## Customization

- **Model**: You can experiment with different pre-trained models available in TensorFlow/Keras for feature extraction.
- **Threshold**: Adjust the threshold value based on the desired sensitivity of similarity detection.

## Contributions

Contributions are welcome! If you have any suggestions, enhancements, or bug fixes, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
