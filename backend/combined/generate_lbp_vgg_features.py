import pandas as pd

lbp_df = pd.read_csv('../lbp/new_images_lbp_features.csv')
vgg_df = pd.read_csv('../cnn/new_images_deep_features.csv')

merged_df = pd.merge(lbp_df, vgg_df, on='Filename')
merged_df.to_csv('images_lbp_vgg_features.csv', index=False)
