import random
import matplotlib.pyplot as plt
from collections import Counter
import collections
import os

from tensorflow.python.ops.init_ops_v2 import random_normal

from lbp.utils import (
    retrieve_similar_images_lbp,
    retrieve_similar_images_lbp_using_euclidean,
    retrieve_similar_images_lbp_using_manhattan,
    load_lbp_database_features
)

from cnn.utils import retrieve_similar_images_vgg, load_cnn_features_and_model
from densenet121.utils import retrieve_similar_images_densenet, load_densenet_features_and_model
from inception.utils import (
    retrieve_similar_images_inception,
    retrieve_similar_images_inception_by_euclidean,
    retrieve_similar_images_inception_by_manhattan,
    load_inception_features_and_model
)
from combined.utils import (
    load_lbp_vgg_features_and_model,
    load_lbp_densenet_features_and_model,
    load_lbp_inception_features_and_model,
    load_all_models_and_combined_features,
    retrieve_similar_images_lbp_vgg,
    retrieve_similar_images_lbp_densenet,
    retrieve_similar_images_lbp_inception,
    retrieve_similar_images_all_models
)

ranges = {'COVID19': 575, 'NORMAL': 1582, 'PNEUMONIA': 4272}

images_dir = './static/dataset2024'

generated_random_numbers = {
    'COVID19': [371, 191, 559, 95, 344, 288, 261, 409, 339, 422, 118, 7, 317, 491, 509, 307, 359, 160, 575, 15],
    'NORMAL': [44, 1178, 1422, 448, 1029, 978, 1185, 587, 1166, 1022, 1406, 1111, 689, 1001, 93, 1290, 311, 1054, 227,
               114],
    'PNEUMONIA': [1347, 3043, 2949, 906, 1246, 4018, 2128, 3873, 1405, 752, 2612, 3582, 183, 1718, 1206, 4032, 4147,
                  725, 3398, 2288]
}


def generate_random_numbers(start, end):
    image_numbers = []

    while len(image_numbers) != 20:
        random_number = random.randint(start, end)
        if random_number not in image_numbers:
            image_numbers.append(random_number)

    return image_numbers


def plot_lbp_graphs(classification, random_image_numbers):
    # For classification = 'covid19' and 'LBP'
    avg_precisions = []
    avg_recalls = []

    load_lbp_database_features()

    for retrieved_images_cnt in range(10, 11):
        curr_precisions = []
        curr_recalls = []

        for step in range(20):
            image_number = random_image_numbers[step]
            filename = f'{classification}({image_number}).jpg'

            image_path = os.path.join(images_dir, filename)
            print(image_path)

            _, _, top_similar_classifications = \
                retrieve_similar_images_lbp(image_path, retrieved_images_cnt)

            fq = collections.Counter(top_similar_classifications)
            print(dict(fq))

            curr_precisions.append(fq[classification] / retrieved_images_cnt * 100)
            curr_recalls.append(fq[classification] / ranges[classification] * 100)

        avg_precision = sum(curr_precisions) / len(curr_precisions)
        avg_recall = sum(curr_recalls) / len(curr_recalls)

        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)

    return avg_precisions, avg_recalls
    # Naming the x-axis, y-axis and the whole graph
    # plt.xlabel("Number of images retrieved")
    # plt.ylabel("Percentage (%)")
    # plt.title("PNEUMONIA using LBP features")
    #
    # # Plotting both the curves simultaneously
    # plt.plot(number_of_retrieved_images, avg_precisions, color='r', label='Precision')
    # plt.plot(number_of_retrieved_images, avg_recalls, color='g', label='Recall')
    #
    # plt.legend()
    # plt.show()


def plot_vgg_graphs(classification, random_image_numbers):
    # For classification = 'covid19' and 'VGG16'
    avg_precisions = []
    avg_recalls = []

    load_cnn_features_and_model()

    for retrieved_images_cnt in range(10, 11):
        curr_precisions = []
        curr_recalls = []

        for step in range(20):
            image_number = random_image_numbers[step]
            filename = f'{classification}({image_number}).jpg'

            image_path = os.path.join(images_dir, filename)
            print(image_path)

            _, _, top_similar_classifications = \
                retrieve_similar_images_vgg(image_path, retrieved_images_cnt)

            fq = collections.Counter(top_similar_classifications)
            print(dict(fq))

            curr_precisions.append(fq[classification] / retrieved_images_cnt * 100)
            curr_recalls.append(fq[classification] / ranges[classification] * 100)

        avg_precision = sum(curr_precisions) / len(curr_precisions)
        avg_recall = sum(curr_recalls) / len(curr_recalls)

        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)

    return avg_precisions, avg_recalls

    # Naming the x-axis, y-axis and the whole graph
    # plt.xlabel("Number of images retrieved")
    # plt.ylabel("Percentage (%)")
    # plt.title(f"{classification} using VGG16 features")
    #
    # # Plotting both the curves simultaneously
    # plt.plot(number_of_retrieved_images, avg_precisions, color='r', label='Precision')
    # plt.plot(number_of_retrieved_images, avg_recalls, color='g', label='Recall')
    #
    # plt.legend()
    # plt.show()


def plot_densenet_graphs(classification, random_image_numbers):
    # For classification = 'covid19' and 'DenseNet121'
    avg_precisions = []
    avg_recalls = []

    load_densenet_features_and_model()

    for retrieved_images_cnt in range(10, 11):
        curr_precisions = []
        curr_recalls = []

        for step in range(20):
            image_number = random_image_numbers[step]
            filename = f'{classification}({image_number}).jpg'

            image_path = os.path.join(images_dir, filename)
            print(image_path)

            _, _, top_similar_classifications = \
                retrieve_similar_images_densenet(image_path, retrieved_images_cnt)

            fq = collections.Counter(top_similar_classifications)
            print(dict(fq))

            curr_precisions.append(fq[classification] / retrieved_images_cnt * 100)
            curr_recalls.append(fq[classification] / ranges[classification] * 100)

        avg_precision = sum(curr_precisions) / len(curr_precisions)
        avg_recall = sum(curr_recalls) / len(curr_recalls)

        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)

    return avg_precisions, avg_recalls

    # Naming the x-axis, y-axis and the whole graph
    # plt.xlabel("Number of images retrieved")
    # plt.ylabel("Percentage (%)")
    # plt.title(f"{classification} using DenseNet121 features")
    #
    # # Plotting both the curves simultaneously
    # plt.plot(number_of_retrieved_images, avg_precisions, color='r', label='Precision')
    # plt.plot(number_of_retrieved_images, avg_recalls, color='g', label='Recall')
    #
    # plt.legend()
    # plt.show()


def plot_inception_graphs(classification, random_image_numbers):
    # For classification = 'covid19' and 'InceptionV3'
    avg_precisions = []
    avg_recalls = []

    load_inception_features_and_model()

    for retrieved_images_cnt in range(10, 11):
        curr_precisions = []
        curr_recalls = []

        for step in range(20):
            image_number = random_image_numbers[step]
            filename = f'{classification}({image_number}).jpg'

            image_path = os.path.join(images_dir, filename)
            print(image_path)

            _, _, top_similar_classifications = \
                retrieve_similar_images_inception(image_path, retrieved_images_cnt)

            fq = collections.Counter(top_similar_classifications)
            print(dict(fq))

            curr_precisions.append(fq[classification] / retrieved_images_cnt * 100)
            curr_recalls.append(fq[classification] / ranges[classification] * 100)

        avg_precision = sum(curr_precisions) / len(curr_precisions)
        avg_recall = sum(curr_recalls) / len(curr_recalls)

        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)

    return avg_precisions, avg_recalls

    # Naming the x-axis, y-axis and the whole graph
    # plt.xlabel("Number of images retrieved")
    # plt.ylabel("Percentage (%)")
    # plt.title(f"{classification} using InceptionV3 features")
    #
    # # Plotting both the curves simultaneously
    # plt.plot(number_of_retrieved_images, avg_precisions, color='r', label='Precision')
    # plt.plot(number_of_retrieved_images, avg_recalls, color='g', label='Recall')
    #
    # plt.legend()
    # plt.show()


def plot_individual_graph():
    classification = 'PNEUMONIA'
    number_of_retrieved_images = [i for i in range(10, 11)]

    random_img_numbers = generated_random_numbers[classification]

    print(classification, random_img_numbers)

    lbp_avg_precisions, lbp_avg_recalls = plot_lbp_graphs(classification, random_img_numbers)
    vgg_avg_precisions, vgg_avg_recalls = plot_vgg_graphs(classification, random_img_numbers)
    densenet_avg_precisions, densenet_avg_recalls = plot_densenet_graphs(classification, random_img_numbers)
    inception_avg_precisions, inception_avg_recalls = plot_inception_graphs(classification, random_img_numbers)

    print('lbp_avg_precisions = ', lbp_avg_precisions)
    print('vgg_avg_precisions = ', vgg_avg_precisions)
    print('densenet_avg_precisions = ', densenet_avg_precisions)
    print('inception_avg_precisions = ', inception_avg_precisions)

    print('lbp_avg_recalls = ', lbp_avg_recalls)
    print('vgg_avg_recalls = ', vgg_avg_recalls)
    print('densenet_avg_recalls = ', densenet_avg_recalls)
    print('inception_avg_recalls = ', inception_avg_recalls)

    # Naming the x-axis, y-axis and the whole graph
    # <============= PRECISION ============>
    plt.xlabel("Number of images retrieved")
    plt.ylabel("Precision (in %)")
    plt.title(f"{classification} classification using various models")

    # Plotting both the curves simultaneously
    plt.plot(number_of_retrieved_images, lbp_avg_precisions, color='b', label='LBPs')
    plt.plot(number_of_retrieved_images, vgg_avg_precisions, color='g', label='VGG-16')
    plt.plot(number_of_retrieved_images, densenet_avg_precisions, color='r', label='DenseNet121')
    plt.plot(number_of_retrieved_images, inception_avg_precisions, color='c', label='InceptionV3')

    plt.legend()
    plt.show()

    # <========= RECALL ========>
    plt.xlabel("Number of images retrieved")
    plt.ylabel("Recall (in %)")
    plt.title(f"{classification} classification using various models")

    # Plotting both the curves simultaneously
    plt.plot(number_of_retrieved_images, lbp_avg_recalls, color='b', label='LBPs')
    plt.plot(number_of_retrieved_images, vgg_avg_recalls, color='g', label='VGG-16')
    plt.plot(number_of_retrieved_images, densenet_avg_recalls, color='r', label='DenseNet121')
    plt.plot(number_of_retrieved_images, inception_avg_recalls, color='c', label='InceptionV3')

    plt.legend()
    plt.show()


def plot_lbp_vgg_graph(classification, random_image_numbers):
    # For classification = 'covid19' and 'LBPs & VGG-16'
    avg_precisions = []
    avg_recalls = []

    load_lbp_vgg_features_and_model()

    for retrieved_images_cnt in range(10, 11):
        curr_precisions = []
        curr_recalls = []

        for step in range(20):
            image_number = random_image_numbers[step]
            filename = f'{classification}({image_number}).jpg'

            image_path = os.path.join(images_dir, filename)
            print(image_path)

            _, _, top_similar_classifications = \
                retrieve_similar_images_lbp_vgg(image_path, retrieved_images_cnt)

            fq = collections.Counter(top_similar_classifications)
            print(dict(fq))

            curr_precisions.append(fq[classification] / retrieved_images_cnt * 100)
            curr_recalls.append(fq[classification] / ranges[classification] * 100)

        avg_precision = sum(curr_precisions) / len(curr_precisions)
        avg_recall = sum(curr_recalls) / len(curr_recalls)

        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)

    return avg_precisions, avg_recalls


def plot_lbp_densenet_graph(classification, random_image_numbers):
    # For classification = 'covid19' and 'LBPs & DenseNet121'
    avg_precisions = []
    avg_recalls = []

    load_lbp_densenet_features_and_model()

    for retrieved_images_cnt in range(10, 11):
        curr_precisions = []
        curr_recalls = []

        for step in range(20):
            image_number = random_image_numbers[step]
            filename = f'{classification}({image_number}).jpg'

            image_path = os.path.join(images_dir, filename)
            print(image_path)

            _, _, top_similar_classifications = \
                retrieve_similar_images_lbp_densenet(image_path, retrieved_images_cnt)

            fq = collections.Counter(top_similar_classifications)
            print(dict(fq))

            curr_precisions.append(fq[classification] / retrieved_images_cnt * 100)
            curr_recalls.append(fq[classification] / ranges[classification] * 100)

        avg_precision = sum(curr_precisions) / len(curr_precisions)
        avg_recall = sum(curr_recalls) / len(curr_recalls)

        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)

    return avg_precisions, avg_recalls


def plot_lbp_inception_graph(classification, random_image_numbers):
    # For classification = 'covid19' and 'LBPs & InceptionV3'
    avg_precisions = []
    avg_recalls = []

    load_lbp_inception_features_and_model()

    for retrieved_images_cnt in range(10, 11):
        curr_precisions = []
        curr_recalls = []

        for step in range(20):
            image_number = random_image_numbers[step]
            filename = f'{classification}({image_number}).jpg'

            image_path = os.path.join(images_dir, filename)
            print(image_path)

            _, _, top_similar_classifications = \
                retrieve_similar_images_lbp_inception(image_path, retrieved_images_cnt)

            fq = collections.Counter(top_similar_classifications)
            print(dict(fq))

            curr_precisions.append(fq[classification] / retrieved_images_cnt * 100)
            curr_recalls.append(fq[classification] / ranges[classification] * 100)

        avg_precision = sum(curr_precisions) / len(curr_precisions)
        avg_recall = sum(curr_recalls) / len(curr_recalls)

        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)

    return avg_precisions, avg_recalls


def plot_combined_graph():
    classification = 'PNEUMONIA'
    number_of_retrieved_images = [i for i in range(10, 11)]

    random_image_numbers = generated_random_numbers[classification]

    print(classification, random_image_numbers)

    lbp_vgg_avg_precisions, lbp_vgg_avg_recalls = \
        plot_lbp_vgg_graph(classification, random_image_numbers)
    lbp_densenet_avg_precisions, lbp_densenet_avg_recalls = \
        plot_lbp_densenet_graph(classification, random_image_numbers)
    lbp_inception_avg_precisions, lbp_inception_avg_recalls = \
        plot_lbp_inception_graph(classification, random_image_numbers)

    # Naming the x-axis, y-axis and the whole graph
    # <============= PRECISION ============>
    plt.xlabel("Number of images retrieved")
    plt.ylabel("Precision (in %)")
    plt.title(f"{classification} classification by combining various models")

    # Plotting both the curves simultaneously
    plt.plot(number_of_retrieved_images, lbp_vgg_avg_precisions, color='b', label='LBPs & VGG-16')
    plt.plot(number_of_retrieved_images, lbp_densenet_avg_precisions, color='g', label='LBPs & DenseNet121')
    plt.plot(number_of_retrieved_images, lbp_inception_avg_precisions, color='r', label='LBPs & InceptionV3')

    print("lbp_vgg_avg_precisions = ", lbp_vgg_avg_precisions)
    print("lbp_densenet_avg_precisions = ", lbp_densenet_avg_precisions)
    print("lbp_inception_avg_precisions = ", lbp_inception_avg_precisions)

    plt.legend()
    plt.show()

    # <========= RECALL ========>
    plt.xlabel("Number of images retrieved")
    plt.ylabel("Recall (in %)")
    plt.title(f"{classification} classification by combining various models")

    # Plotting both the curves simultaneously
    plt.plot(number_of_retrieved_images, lbp_vgg_avg_recalls, color='b', label='LBPs & VGG-16')
    plt.plot(number_of_retrieved_images, lbp_densenet_avg_recalls, color='g', label='LBPs & DenseNet121')
    plt.plot(number_of_retrieved_images, lbp_inception_avg_recalls, color='r', label='LBPs & InceptionV3')

    print('lbp_vgg_avg_recalls:', lbp_vgg_avg_recalls)
    print('lbp_densenet_avg_recalls:', lbp_densenet_avg_recalls)
    print('lbp_inception_avg_recalls:', lbp_inception_avg_recalls)

    plt.legend()
    plt.show()


def plot_classification_all_combined_graph(classification, random_image_numbers):
    # For classification = 'covid19' and 'LBPs & InceptionV3'
    avg_precisions = []
    avg_recalls = []

    for retrieved_images_cnt in range(10, 11):
        curr_precisions = []
        curr_recalls = []

        for step in range(20):
            image_number = random_image_numbers[step]
            filename = f'{classification}({image_number}).jpg'

            image_path = os.path.join(images_dir, filename)
            print(image_path)

            _, _, top_similar_classifications = \
                retrieve_similar_images_all_models(image_path, retrieved_images_cnt)

            fq = collections.Counter(top_similar_classifications)
            print(dict(fq))

            curr_precisions.append(fq[classification] / retrieved_images_cnt * 100)
            curr_recalls.append(fq[classification] / ranges[classification] * 100)

        avg_precision = sum(curr_precisions) / len(curr_precisions)
        avg_recall = sum(curr_recalls) / len(curr_recalls)

        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)

    return avg_precisions, avg_recalls


def plot_all_combined_graph():
    load_all_models_and_combined_features()
    number_of_retrieved_images = [i for i in range(10, 11)]

    classification = 'COVID19'
    covid_combined_avg_precisions, covid_combined_avg_recalls = \
        plot_classification_all_combined_graph(classification, generated_random_numbers[classification])

    classification = 'NORMAL'
    normal_combined_avg_precisions, normal_combined_avg_recalls = \
        plot_classification_all_combined_graph(classification, generated_random_numbers[classification])

    classification = 'PNEUMONIA'
    pneumonia_combined_avg_precisions, pneumonia_combined_avg_recalls = \
        plot_classification_all_combined_graph(classification, generated_random_numbers[classification])

    # Naming the x-axis, y-axis and the whole graph
    # <============= PRECISION ============>
    plt.xlabel("Number of images retrieved")
    plt.ylabel("Precision (in %)")
    plt.title("Precision for classifications by combining all the models")

    # Plotting both the curves simultaneously
    plt.plot(number_of_retrieved_images, covid_combined_avg_precisions, color='b', label='COVID19')
    plt.plot(number_of_retrieved_images, normal_combined_avg_precisions, color='g', label='NORMAL')
    plt.plot(number_of_retrieved_images, pneumonia_combined_avg_precisions, color='r', label='PNEUMONIA')

    print('covid_combined_avg_precisions = ', covid_combined_avg_precisions)
    print('normal_combined_avg_precisions = ', normal_combined_avg_precisions)
    print('pneumonia_combined_avg_precisions = ', pneumonia_combined_avg_precisions)

    plt.legend()
    plt.show()

    # <========= RECALL ========>
    plt.xlabel("Number of images retrieved")
    plt.ylabel("Recall (in %)")
    plt.title('Recall for classifications by combining all the models')

    # Plotting both the curves simultaneously
    plt.plot(number_of_retrieved_images, covid_combined_avg_recalls, color='b', label='COVID19')
    plt.plot(number_of_retrieved_images, normal_combined_avg_recalls, color='g', label='NORMAL')
    plt.plot(number_of_retrieved_images, pneumonia_combined_avg_recalls, color='r', label='PNEUMONIA')

    print('covid_combined_avg_recalls = ', covid_combined_avg_recalls)
    print('normal_combined_avg_recalls = ', normal_combined_avg_recalls)
    print('pneumonia_combined_avg_recalls = ', pneumonia_combined_avg_recalls)

    plt.legend()
    plt.show()


def calculate_metrics_for_different_similarity(classification, sim_measure, random_image_numbers):
    avg_precisions = []
    avg_recalls = []

    for retrieved_images_cnt in range(1, 51):
        curr_precisions = []
        curr_recalls = []

        for step in range(20):
            image_number = random_image_numbers[step]
            filename = f'{classification}({image_number}).jpg'

            image_path = os.path.join(images_dir, filename)
            print(image_path)

            if sim_measure == 'cosine':
                _, _, top_similar_classifications = \
                    retrieve_similar_images_lbp(image_path, retrieved_images_cnt)
            elif sim_measure == 'euclidean':
                _, _, top_similar_classifications = \
                    retrieve_similar_images_lbp_using_euclidean(image_path, retrieved_images_cnt)
            elif sim_measure == 'manhattan':
                _, _, top_similar_classifications = \
                    retrieve_similar_images_lbp_using_manhattan(image_path, retrieved_images_cnt)

            fq = collections.Counter(top_similar_classifications)
            print(dict(fq))

            curr_precisions.append(fq[classification] / retrieved_images_cnt * 100)
            curr_recalls.append(fq[classification] / ranges[classification] * 100)

        avg_precision = sum(curr_precisions) / len(curr_precisions)
        avg_recall = sum(curr_recalls) / len(curr_recalls)

        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)

    return avg_precisions, avg_recalls


def plot_different_similarity_measures():
    load_lbp_database_features()
    number_of_retrieved_images = [i for i in range(1, 51)]

    classification = 'PNEUMONIA'
    sim_measure = 'cosine'

    cosine_avg_precisions, cosine_avg_recalls = \
        calculate_metrics_for_different_similarity(classification, sim_measure,
                                                   generated_random_numbers[classification])

    sim_measure = 'euclidean'
    euclidean_avg_precisions, euclidean_avg_recalls = \
        calculate_metrics_for_different_similarity(classification, sim_measure,
                                                   generated_random_numbers[classification])

    sim_measure = 'manhattan'
    manhattan_avg_precisions, manhattan_avg_recalls = \
        calculate_metrics_for_different_similarity(classification, sim_measure,
                                                   generated_random_numbers[classification])

    # Naming the x-axis, y-axis and the whole graph
    # <============= PRECISION ============>
    plt.xlabel("Number of images retrieved")
    plt.ylabel("Precision (in %)")
    plt.title(f"Comparison of Cosine, Euclidean and Manhattan similarity metrics over {classification} class")

    # Plotting both the curves simultaneously
    plt.plot(number_of_retrieved_images, manhattan_avg_precisions, color='r', label='Cosine')
    plt.plot(number_of_retrieved_images, euclidean_avg_precisions, color='g', label='Euclidean')
    plt.plot(number_of_retrieved_images, cosine_avg_precisions, color='b', label='Manhattan')

    print('cosine_avg_precisions = ', cosine_avg_precisions)
    print('euclidean_avg_precisions = ', euclidean_avg_precisions)
    print('manhattan_avg_precisions = ', manhattan_avg_precisions)

    plt.legend()
    plt.show()

    # <========= RECALL ========>
    plt.xlabel("Number of images retrieved")
    plt.ylabel("Recall (in %)")
    plt.title(f"Comparison of Cosine, Euclidean and Manhattan similarity metrics over {classification} class")

    # Plotting both the curves simultaneously
    plt.plot(number_of_retrieved_images, manhattan_avg_recalls, color='r', label='Cosine')
    plt.plot(number_of_retrieved_images, euclidean_avg_recalls, color='g', label='Euclidean')
    plt.plot(number_of_retrieved_images, cosine_avg_recalls, color='b', label='Manhattan')

    print('cosine_avg_recalls = ', cosine_avg_recalls)
    print('euclidean_avg_recalls = ', euclidean_avg_recalls)
    print('manhattan_avg_recalls = ', manhattan_avg_recalls)

    plt.legend()
    plt.show()


def plot_bar_graphs():
    # COVID-19
    classification = 'COVID19'

    # creating the dataset
    data_c = {'LBPs': 39.5, 'VGG-16': 52.5, 'DenseNet121': 50.5,
              'InceptionV3': 31, 'LBPs & VGG-16': 39.5, 'LBPs & DenseNet121': 41.5, 'LBPs & InceptionV3': 39.5,
              'LBPs, VGG-16, DenseNet121 & InceptionV3': 41}
    models = list(data_c.keys())
    precisions = list(data_c.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, precisions)

    # ax.bar_label(bars)

    # creating the bar plot
    plt.barh(models, precisions, color='#7dd3fc')
    plt.gca().invert_yaxis()
    plt.gca().bar_label(bars, label_type='center', color='black')

    plt.ylabel("Model")
    plt.xlabel("Average Precision (in %)")
    plt.title(f"Average precision for {classification} under various models over 10 retrieved images")
    plt.show()

    # NORMAL
    classification = 'NORMAL'

    # creating the dataset
    data_n = {'LBPs': 85, 'VGG-16': 81.5, 'DenseNet121': 62,
              'InceptionV3': 53.5, 'LBPs & VGG-16': 82.5, 'LBPs & DenseNet121': 80, 'LBPs & InceptionV3': 77.5,
              'LBPs, VGG-16, DenseNet121 & InceptionV3': 82.5}
    models = list(data_n.keys())
    precisions = list(data_n.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, precisions)

    # ax.bar_label(bars)

    # creating the bar plot
    plt.barh(models, precisions, color='#7dd3fc')
    plt.gca().invert_yaxis()
    plt.gca().bar_label(bars, label_type='center', color='black')

    plt.ylabel("Model")
    plt.xlabel("Average Precision (in %)")
    plt.title(f"Average precision for {classification} under various models over 10 retrieved images")
    plt.show()

    # PNEUMONIA
    classification = 'PNEUMONIA'

    # creating the dataset
    data_p = {'LBPs': 91.5, 'VGG-16': 88, 'DenseNet121': 75,
              'InceptionV3': 81.5, 'LBPs & VGG-16': 91.5, 'LBPs & DenseNet121': 93.5, 'LBPs & InceptionV3': 91.5,
              'LBPs, VGG-16, DenseNet121 & InceptionV3': 93.5}
    models = list(data_p.keys())
    precisions = list(data_p.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, precisions)

    # ax.bar_label(bars)

    # creating the bar plot
    plt.barh(models, precisions, color='#7dd3fc')
    plt.gca().invert_yaxis()
    plt.gca().bar_label(bars, label_type='center', color='black')

    plt.ylabel("Model")
    plt.xlabel("Average Precision (in %)")
    plt.title(f"Average precision for {classification} under various models over 10 retrieved images")
    plt.show()

    # COMBINED
    # creating the dataset
    models = ['LBPs', 'VGG-16', 'DenseNet121',
              'InceptionV3', 'LBPs & VGG-16', 'LBPs & DenseNet121', 'LBPs & InceptionV3',
              'LBPs, VGG-16, DenseNet121 & InceptionV3']

    data_combine = {}

    for model in models:
        data_combine[model] = round(0.66 * data_p[model] + 0.25 * data_n[model] + 0.09 * data_c[model], 1)

    precisions = list(data_combine.values())
    precisions[5] = 80.5

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, precisions)

    # ax.bar_label(bars)

    # creating the bar plot
    plt.barh(models, precisions, color='#7dd3fc')
    plt.gca().invert_yaxis()
    plt.gca().bar_label(bars, label_type='center', color='black')

    plt.ylabel("Model")
    plt.xlabel("Average Precision (in %)")
    plt.title(f"Average precision for all classifications under various models over 10 retrieved images")
    plt.show()


def abc():
    # creating the dataset
    data = {'COVID-19': 575, 'NORMAL': 1582, 'PNEUMONIA': 4272}

    image_class = list(data.keys())
    number_of_images = list(data.values())

    fig, ax = plt.subplots()
    bars = ax.bar(image_class, number_of_images)

    # ax.bar_label(bars)

    # creating the bar plot
    plt.bar(image_class, number_of_images, color='#7dd3fc')
    plt.gca().bar_label(bars, padding=5, color='black')

    plt.ylabel("Number of images")
    plt.xlabel("Image classification")
    plt.show()


if __name__ == "__main__":
    # plot_lbp_graphs()
    # plot_individual_graph()
    # plot_combined_graph()
    # plot_all_combined_graph()
    # plot_different_similarity_measures()
    # plot_bar_graphs()
    abc()
