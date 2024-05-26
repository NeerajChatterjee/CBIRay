from keras.utils import load_img, img_to_array
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import cv2
import collections
from backend.utils.find_classification import find_image_classification
from backend.utils.create_model import create_densenet121_model

features_df = pd.DataFrame([])
query_image = None
query_image_file_path = None
denseNet121_model = None
times = 1


def load_densenet_features_and_model():
    s_time = time.time()
    global features_df, denseNet121_model

    # Load the features from the CSV file
    features_df = pd.read_csv('./densenet121/images_densenet_features.csv') # path wrt to `main.py`

    e_time = time.time()  # ~2 seconds
    print("DenseNet121 features loaded in time: ", e_time - s_time)

    s_time = time.time()

    # Load the VGG model
    denseNet121_model = create_densenet121_model()

    e_time = time.time()
    print("DenseNet121 created in time: ", e_time - s_time)


# Function to extract features from an image
def extract_query_features(img_path, model):
    if model is None:
        return

    img = load_img(img_path, target_size=(224, 224), color_mode="grayscale")
    input_arr = img_to_array(img)
    merged_input_arr = cv2.merge((input_arr, input_arr, input_arr)) # Converting to (224 x 224 x 3) , i.e., 3 channels
    img_array = np.array([merged_input_arr])  # Convert single image to a batch.
    img_features = model.predict(img_array)

    return img_features[0]


def retrieve_similar_images_densenet(image_path, images_count=20):
    query_image_features = extract_query_features(image_path, denseNet121_model)

    # Remove the 'Filename' column for comparison
    stored_features = features_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_image_features], stored_features)[0]

    top_n = int(images_count)

    # Get indices of top `n` most similar images
    top_similar_indices = similarities.argsort()[-top_n:][::-1]

    # Retrieve top 10 similar filenames and their similarity values
    top_similar_filenames = features_df.iloc[top_similar_indices]['Filename'].values
    top_similar_values = similarities[top_similar_indices]
    top_similar_classifications = []

    # Print the top n most similar filenames and their similarity values
    for idx, (filename, sim_value) in enumerate(zip(top_similar_filenames, top_similar_values), 1):
        top_similar_classifications.append(find_image_classification(filename))
        # print(f"{idx}. {filename} - Similarity: {sim_value:.4f}")

    fq = collections.Counter(top_similar_classifications)
    print(dict(fq))

    return [top_similar_filenames, top_similar_values, top_similar_classifications]

