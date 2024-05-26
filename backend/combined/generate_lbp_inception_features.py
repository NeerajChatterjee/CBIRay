import pandas as pd

lbp_df = pd.read_csv('../lbp/new_images_lbp_features.csv')
inception_df = pd.read_csv('../inception/images_inception_features.csv')

merged_df = pd.merge(lbp_df, inception_df, on='Filename')
print(merged_df.shape)
merged_df.to_csv('images_lbp_inception_features.csv', index=False)
