def find_image_classification(filename):
    if 'COVID19' in filename:
        return 'COVID19'
    elif 'PNEUMONIA' in filename:
        return 'PNEUMONIA'
    elif 'NORMAL' in filename:
        return 'NORMAL'
    else:
        return 'UNCLASSIFIED'

