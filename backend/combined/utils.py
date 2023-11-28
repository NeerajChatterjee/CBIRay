import time

import numpy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import collections
from backend.utils.find_classification import find_image_classification
from backend.utils.create_model import create_vgg_model
from backend.cnn.utils import extract_features as extract_deep_features
from backend.lbp.utils import compute_lbp_features, create_histogram

features_df = pd.DataFrame([])
query_image = None
query_image_file_path = None
vgg_model = None
times = 1


def load_combined_features_and_model():
    s_time = time.time()
    global features_df, vgg_model

    # Load the features from the CSV file
    features_df = pd.read_csv('./combined/images_combined_features.csv')

    e_time = time.time()  # ~2 seconds
    print("CNN features loaded in time: ", e_time - s_time)

    s_time = time.time()

    # Load the VGG model
    vgg_model = create_vgg_model()

    e_time = time.time()
    print("VGG model created in time: ", e_time - s_time)


def retrieve_similar_images_combined(image_path, images_count):
    query_features_vgg = extract_deep_features(image_path, vgg_model)
    query_features_lbp = numpy.asarray(create_histogram(compute_lbp_features(image_path)))

    query_features_combined = np.append(query_features_vgg, query_features_lbp, axis=0)
    print(len(query_features_combined))

    # Remove the 'Filename' column for comparison
    stored_features = features_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_features_combined], stored_features)[0]

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