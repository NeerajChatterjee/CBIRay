import pandas as pd

lbp_df = pd.read_csv('../lbp/new_images_lbp_features.csv')
densenet_df = pd.read_csv('../densenet121/images_densenet_features.csv')

merged_df = pd.merge(lbp_df, densenet_df, on='Filename')
print(merged_df.shape)
merged_df.to_csv('images_lbp_densenet_features.csv', index=False)
