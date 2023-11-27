def find_image_classification(filename):
    print(type(filename))
    print(filename)
    if 'NORMAL' in filename or filename[0:2] == 'IM':
        return 'NORMAL'
    elif 'person' in filename:
        return 'PNEUMONIA'
    else:
        return 'UNCLASSIFIED'
