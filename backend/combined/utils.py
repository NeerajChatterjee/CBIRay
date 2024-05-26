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
from backend.densenet121.utils import extract_query_features as extract_densenet_features, create_densenet121_model
from backend.inception.utils import extract_query_features as extract_inception_features, create_inception_model

features_lbp_vgg_df = pd.DataFrame([])
features_lbp_densenet_df = pd.DataFrame([])
features_lbp_inception_df = pd.DataFrame([])
features_all_df = pd.DataFrame([])
query_image = None
query_image_file_path = None
vgg_model = None
densenet_model = None
inception_model = None
times = 1


def load_lbp_vgg_features_and_model():
    s_time = time.time()
    global features_lbp_vgg_df, vgg_model

    # Load the features from the CSV file
    features_lbp_vgg_df = pd.read_csv('./combined/images_lbp_vgg_features.csv')

    e_time = time.time()  # ~2 seconds
    print("LBPs & VGG-16 features loaded in time: ", e_time - s_time)

    s_time = time.time()

    # Load the VGG model
    vgg_model = create_vgg_model()

    e_time = time.time()
    print("VGG model created in time: ", e_time - s_time)


def retrieve_similar_images_lbp_vgg(image_path, images_count):
    query_features_vgg = extract_deep_features(image_path, vgg_model)
    query_features_lbp = numpy.asarray(create_histogram(compute_lbp_features(image_path)))

    query_features_lbp_vgg = np.append(query_features_lbp, query_features_vgg, axis=0)
    print(len(query_features_lbp_vgg))

    # Remove the 'Filename' column for comparison
    stored_features = features_lbp_vgg_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_features_lbp_vgg], stored_features)[0]

    top_n = int(images_count)

    # Get indices of top 10 most similar images
    top_similar_indices = similarities.argsort()[-top_n:][::-1]

    # Retrieve top 10 similar filenames and their similarity values
    top_similar_filenames = features_lbp_vgg_df.iloc[top_similar_indices]['Filename'].values
    top_similar_values = similarities[top_similar_indices]
    top_similar_classifications = []

    # Print the top n most similar filenames and their similarity values
    for idx, (filename, sim_value) in enumerate(zip(top_similar_filenames, top_similar_values), 1):
        top_similar_classifications.append(find_image_classification(filename))
        # print(f"{idx}. {filename} - Similarity: {sim_value:.4f}")

    fq = collections.Counter(top_similar_classifications)
    print(dict(fq))

    return [top_similar_filenames, top_similar_values, top_similar_classifications]


def load_lbp_densenet_features_and_model():
    s_time = time.time()
    global features_lbp_densenet_df, densenet_model

    # Load the features from the CSV file
    features_lbp_densenet_df = pd.read_csv('./combined/images_lbp_densenet_features.csv')

    e_time = time.time()  # ~2 seconds
    print("LBPs & DenseNet121 features loaded in time: ", e_time - s_time)

    s_time = time.time()

    # Load the DenseNet model
    densenet_model = create_densenet121_model()

    e_time = time.time()
    print("DenseNet121 model created in time: ", e_time - s_time)


def retrieve_similar_images_lbp_densenet(image_path, images_count):
    query_features_densenet = extract_densenet_features(image_path, densenet_model)
    query_features_lbp = numpy.asarray(create_histogram(compute_lbp_features(image_path)))

    query_features_lbp_densenet = np.append(query_features_lbp, query_features_densenet, axis=0)
    print(len(query_features_lbp_densenet))

    # Remove the 'Filename' column for comparison
    stored_features = features_lbp_densenet_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_features_lbp_densenet], stored_features)[0]

    top_n = int(images_count)

    # Get indices of top 10 most similar images
    top_similar_indices = similarities.argsort()[-top_n:][::-1]

    # Retrieve top 10 similar filenames and their similarity values
    top_similar_filenames = features_lbp_densenet_df.iloc[top_similar_indices]['Filename'].values
    top_similar_values = similarities[top_similar_indices]
    top_similar_classifications = []

    # Print the top n most similar filenames and their similarity values
    for idx, (filename, sim_value) in enumerate(zip(top_similar_filenames, top_similar_values), 1):
        top_similar_classifications.append(find_image_classification(filename))
        # print(f"{idx}. {filename} - Similarity: {sim_value:.4f}")

    fq = collections.Counter(top_similar_classifications)
    print(dict(fq))

    return [top_similar_filenames, top_similar_values, top_similar_classifications]


def load_lbp_inception_features_and_model():
    s_time = time.time()
    global features_lbp_inception_df, inception_model

    # Load the features from the CSV file
    features_lbp_inception_df = pd.read_csv('./combined/images_lbp_inception_features.csv')

    e_time = time.time()  # ~2 seconds
    print("LBPs & InceptionV3 features loaded in time: ", e_time - s_time)

    s_time = time.time()

    # Load the Inception model
    inception_model = create_inception_model()

    e_time = time.time()
    print("InceptionV3 model created in time: ", e_time - s_time)


def retrieve_similar_images_lbp_inception(image_path, images_count):
    query_features_inception = extract_inception_features(image_path, inception_model)
    query_features_lbp = numpy.asarray(create_histogram(compute_lbp_features(image_path)))

    query_features_lbp_inception = np.append(query_features_lbp, query_features_inception, axis=0)
    print(len(query_features_lbp_inception))

    # Remove the 'Filename' column for comparison
    stored_features = features_lbp_inception_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_features_lbp_inception], stored_features)[0]

    top_n = int(images_count)

    # Get indices of top 10 most similar images
    top_similar_indices = similarities.argsort()[-top_n:][::-1]

    # Retrieve top 10 similar filenames and their similarity values
    top_similar_filenames = features_lbp_inception_df.iloc[top_similar_indices]['Filename'].values
    top_similar_values = similarities[top_similar_indices]
    top_similar_classifications = []

    # Print the top n most similar filenames and their similarity values
    for idx, (filename, sim_value) in enumerate(zip(top_similar_filenames, top_similar_values), 1):
        top_similar_classifications.append(find_image_classification(filename))
        # print(f"{idx}. {filename} - Similarity: {sim_value:.4f}")

    fq = collections.Counter(top_similar_classifications)
    print(dict(fq))

    return [top_similar_filenames, top_similar_values, top_similar_classifications]


def load_all_models_and_combined_features():
    s_time = time.time()
    global features_all_df, vgg_model, densenet_model, inception_model

    # Load the features from the CSV file
    features_all_df = pd.read_csv('./combined/images_all_combined_features.csv')

    e_time = time.time()  # ~2 seconds
    print("All combined features loaded in time: ", e_time - s_time)

    s_time = time.time()

    # Load the VGG model
    vgg_model = create_vgg_model()
    # Lead the DenseNet model
    densenet_model = create_densenet121_model()
    # Load the Inception model
    inception_model = create_inception_model()

    e_time = time.time()
    print("VGG-16, DenseNet121 & InceptionV3 model created in time: ", e_time - s_time)


def retrieve_similar_images_all_models(image_path, images_count):
    query_features_lbp = numpy.asarray(create_histogram(compute_lbp_features(image_path)))
    query_features_vgg = extract_deep_features(image_path, vgg_model)
    query_features_densenet = extract_densenet_features(image_path, densenet_model)
    query_features_inception = extract_inception_features(image_path, inception_model)

    query_features_all_models = np.append(query_features_lbp, query_features_vgg, axis=0)
    query_features_all_models = np.append(query_features_all_models, query_features_densenet, axis=0)
    query_features_all_models = np.append(query_features_all_models, query_features_inception, axis=0)

    print(len(query_features_all_models))

    # Remove the 'Filename' column for comparison
    stored_features = features_all_df.drop(columns=['Filename']).values

    # Calculate cosine similarity between the query image features and stored features
    similarities = cosine_similarity([query_features_all_models], stored_features)[0]

    top_n = int(images_count)

    # Get indices of top 10 most similar images
    top_similar_indices = similarities.argsort()[-top_n:][::-1]

    # Retrieve top 10 similar filenames and their similarity values
    top_similar_filenames = features_all_df.iloc[top_similar_indices]['Filename'].values
    top_similar_values = similarities[top_similar_indices]
    top_similar_classifications = []

    # Print the top n most similar filenames and their similarity values
    for idx, (filename, sim_value) in enumerate(zip(top_similar_filenames, top_similar_values), 1):
        top_similar_classifications.append(find_image_classification(filename))
        # print(f"{idx}. {filename} - Similarity: {sim_value:.4f}")

    fq = collections.Counter(top_similar_classifications)
    print(dict(fq))

    return [top_similar_filenames, top_similar_values, top_similar_classifications]

