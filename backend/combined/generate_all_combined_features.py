import pandas as pd

lbp_df = pd.read_csv('../lbp/new_images_lbp_features.csv')
vgg_df = pd.read_csv('../cnn/new_images_deep_features.csv')
densenet_df = pd.read_csv('../densenet121/images_densenet_features.csv')
inception_df = pd.read_csv('../inception/images_inception_features.csv')

lbp_vgg_densenet_df = pd.merge(pd.merge(lbp_df, vgg_df, on='Filename'), densenet_df, on='Filename')
merged_df= pd.merge(lbp_vgg_densenet_df, inception_df, on='Filename')
print(merged_df.shape)
merged_df.to_csv('images_all_combined_features.csv', index=False)
