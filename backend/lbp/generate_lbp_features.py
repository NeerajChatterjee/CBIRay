import os
import numpy as np
import pandas as pd
from utils import compute_lbp_features, create_histogram

# Path to the directory containing images
images_dir = './../static/dataset2024'

# List to store extracted features and corresponding filenames
features_list = []
filenames_list = []

cnt = 0

# Loop through each image in the directory
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(images_dir, filename)
        print(filename, image_path)
        img_features = compute_lbp_features(image_path)
        image_feature_vector = create_histogram(img_features)

        features_list.append(image_feature_vector)
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
features_df.to_csv('new_images_lbp_features.csv', index=False)

