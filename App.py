import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained ResNet50 model + higher level layers
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Path to the predefined dataset directory
DATASET_PATH = 'Dataset\\images'

# Function to preprocess image and extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features

# Extract features for all images in the dataset
def prepare_dataset(dataset_path):
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    features_list = [extract_features(img_path, model) for img_path in image_files]
    features_list = np.vstack(features_list)
    np.save('features.npy', features_list)
    with open('image_paths.txt', 'w') as f:
        for img_path in image_files:
            f.write("%s\n" % img_path)
    return features_list, image_files

# Finding similar images
def find_similar_images(query_img_path, model, features_list, image_files):
    query_features = extract_features(query_img_path, model)
    similarities = cosine_similarity(query_features, features_list)
    similar_indices = similarities[0].argsort()[-5:][::-1]  # Get top 5 similar images
    similar_images = [image_files[i] for i in similar_indices]
    return similar_images

# User Interface with Streamlit
def main():
    st.title("Fashion Recommendation System")

    # Prepare features if not already done
    if not os.path.exists('features.npy') or not os.path.exists('image_paths.txt'):
        with st.spinner('Processing dataset...'):
            features_list, image_files = prepare_dataset(DATASET_PATH)
            st.success('Dataset processed successfully!')
    else:
        features_list = np.load('features.npy')
        with open('image_paths.txt', 'r') as f:
            image_files = [line.strip() for line in f.readlines()]

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        query_img = Image.open(uploaded_file)
        st.image(query_img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Searching for similar images...")

        with open("uploaded.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        similar_images = find_similar_images("uploaded.jpg", model, features_list, image_files)

        for img_path in similar_images:
            similar_img = Image.open(img_path)
            st.image(similar_img, caption=img_path, use_column_width=True)

if __name__ == "__main__":
    main()
