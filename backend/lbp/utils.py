from skimage.feature import local_binary_pattern
from collections import Counter
import collections
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import cv2
from backend.utils.find_classification import find_image_classification


features_df = pd.DataFrame([])
query_image = None
query_image_file_path = None
times = 1


def load_lbp_database_features():
    s_time = time.time()
    global features_df

    # Load the features from the CSV file
    features_df = pd.read_csv('./lbp/images_lbp_features.csv')

    e_time = time.time()  # 10 minutes
    print(e_time - s_time)


def compute_lbp_features(image_path, points=8, radius=1):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, dsize=(400, 400))
    # Compute LBP features
    lbp = local_binary_pattern(image, points, radius, method="uniform")
    lbp_flatten = lbp.flatten()
    # Flatten the features and return as a 1D array
    return np.array(lbp_flatten)


def create_histogram(fv):
    frequency_fv = Counter(fv)

    hist = []
    for i in range(256):
        if i in frequency_fv.keys():
            hist.append(frequency_fv[i])
        else:
            hist.append(0)
    return hist


def retrieve_similar_images_lbp(image_path, images_count):

    # Extract features for the query image (similar to the previous process)
    query_features = compute_lbp_features(image_path)
    query_image_feature_vector = create_histogram(query_features)

    # Remove the 'Filename' column for comparison
    stored_features = features_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_image_feature_vector], stored_features)[0]

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
        # print(f"{idx}. {filename} - Similarity: {sim_value:.4f}")

    fq = collections.Counter(top_similar_classifications)
    print(dict(fq))

    return [top_similar_filenames, top_similar_values, top_similar_classifications]
