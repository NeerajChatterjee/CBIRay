import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from backend.lbp.utils import compute_lbp_features, create_histogram
from backend.cnn.utils import extract_features as extract_deep_features
import collections
import numpy as np
import numpy
from backend.utils.find_classification import find_image_classification
from backend.cnn.utils import extract_features
from backend.utils.create_model import create_vgg_model

query_image_path = './static/uploads/IM-0001-0001.jpeg'


def retrieve_lbp_similar_images(image_path, img_count):

    # Extract features for the query image (similar to the previous process)
    query_features = compute_lbp_features(image_path)
    query_image_feature_vector = create_histogram(query_features)

    # Remove the 'Filename' column for comparison
    features_df = pd.read_csv('lbp/images_lbp_features.csv')
    stored_features = features_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_image_feature_vector], stored_features)[0]

    top_n = int(img_count)

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

    return fq

    # return [top_similar_filenames, top_similar_values, top_similar_classifications]


def retrieve_deep_similar_images(image_path, img_count):
    query_features = extract_features(image_path, vgg_model)

    features_df = pd.read_csv('cnn/images_deep_features.csv')

    # Remove the 'Filename' column for comparison
    stored_features = features_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_features], stored_features)[0]

    top_n = int(img_count)

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

    return fq


def retrieve_similar_images_combined(image_path, img_count):
    query_features_vgg = extract_deep_features(image_path, vgg_model)
    query_features_lbp = numpy.asarray(create_histogram(compute_lbp_features(image_path)))

    query_features_combined = np.append(query_features_vgg, query_features_lbp, axis=0)
    print(len(query_features_combined))

    features_df = pd.read_csv('combined/images_combined_features.csv')

    # Remove the 'Filename' column for comparison
    stored_features = features_df.drop(columns=['Filename']).values
    # print(stored_features)

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_features_combined], stored_features)[0]

    top_n = int(img_count)

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

    return fq


if __name__ == "__main__" :
    global vgg_model
    vgg_model = create_vgg_model()

    x_points = []
    y_points = []

    for images_count in range(1, 101):
        # Now, retrieve images using LBP
        fq = retrieve_lbp_similar_images(query_image_path, images_count)

        # Now, retrieve images using Deep features
        # fq = retrieve_deep_similar_images(query_image_path, images_count)

        # Now, retrieve images using combined model
        # fq = retrieve_similar_images_combined(query_image_path, images_count)

        precision = (fq['NORMAL'] / images_count) * 100

        print(f"Precision for {images_count}: ", precision)

        x_points.append(images_count)
        y_points.append(precision)

    print(x_points)
    print(y_points)




