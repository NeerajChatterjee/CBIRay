import os
import numpy as np
import pandas as pd
from utils import create_vgg_model, extract_features

# First we'll train the model without Fine-tuning
vgg_model = create_vgg_model()

# Path to the directory containing images
images_dir = './../static/chest_xray'

# List to store extracted features and corresponding filenames
features_list = []
filenames_list = []

cnt = 0

# Loop through each image in the directory
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(images_dir, filename)
        img_features = extract_features(image_path, vgg_model)
        features_list.append(img_features)
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
features_df.to_csv('images_deep_features.csv', index=False)