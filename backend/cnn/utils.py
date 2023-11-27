from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras import layers
from keras import models
from backend.utils.find_classification import find_image_classification

features_df = pd.DataFrame([])
query_image = None
query_image_file_path = None
vgg_model = None
times = 1


def create_vgg_model():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(240, 240, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))

    model.load_weights('./cnn/vgg_weights.h5')

    return model


def load_cnn_features_and_model():
    s_time = time.time()
    global features_df, vgg_model

    # Load the features from the CSV file
    features_df = pd.read_csv('./cnn/images_deep_features.csv')

    e_time = time.time()  # ~2 seconds
    print("CNN features loaded in time: ", e_time - s_time)

    s_time = time.time()

    # Load the VGG model
    vgg_model = create_vgg_model()

    e_time = time.time()
    print("VGG model created in time: ", e_time - s_time)


# Function to extract features from an image
def extract_features(img_path, model):
    if model is None:
        return

    img = image.load_img(img_path, target_size=(240, 240))  # VGG16 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)

    # arr_features = np.array(features)

    flatten_features = features.flatten()

    return flatten_features


def retrieve_similar_images_vgg(image_path, images_count):

    query_features = extract_features(image_path, vgg_model)
    # print("=================================================================")
    # print(query_features)
    # print("=================================================================")

    # Remove the 'Filename' column for comparison
    stored_features = features_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_features], stored_features)[0]

    top_n = int(images_count)

    # Get indices of top 10 most similar images
    top_similar_indices = similarities.argsort()[-top_n:][::-1]

    # Retrieve top 10 similar filenames and their similarity values
    top_similar_filenames = features_df.iloc[top_similar_indices]['Filename'].values
    top_similar_values = similarities[top_similar_indices]
    top_similar_classifications = []

    # Print the top n most similar filenames and their similarity values
    for idx, (filename, sim_value) in enumerate(zip(top_similar_filenames, top_similar_values), 1):
        top_similar_classifications.append(find_image_classification(filename))
        print(f"{idx}. {filename} - Similarity: {sim_value:.4f}")

    return [top_similar_filenames, top_similar_values, top_similar_classifications]
