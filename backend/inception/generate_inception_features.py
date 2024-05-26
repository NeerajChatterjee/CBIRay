import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import load_img, img_to_array
import cv2
from backend.utils.create_model import create_inception_model

dataset_dir = '../dataset2024'

# List to store extracted features and corresponding filenames
features_list = []
filenames_list = []


def extract_features():
    model = create_inception_model()
    cnt = 0

    # Loop through each image in the directory
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(dataset_dir, filename)
            img = load_img(image_path, target_size=(224, 224), color_mode="grayscale")
            input_arr = img_to_array(img)
            merged_input_arr = cv2.merge((input_arr, input_arr, input_arr))
            img_array = np.array([merged_input_arr])  # Convert single image to a batch.
            img_features = model.predict(img_array)
            features_list.append(img_features[0])
            filenames_list.append(filename)
            print(filename, cnt)
            cnt += 1

    # Convert lists to numpy arrays
    features_array = np.array(features_list)
    filenames_array = np.array(filenames_list)

    # Create a DataFrame to store features and filenames
    features_df = pd.DataFrame(features_array)
    features_df['Filename'] = filenames_array

    # Save features to a CSV file
    features_df.to_csv('images_inception_features.csv', index=False)


if __name__ == '__main__':
    extract_features()