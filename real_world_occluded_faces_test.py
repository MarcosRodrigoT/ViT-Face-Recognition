import os


def remove_empty_directories():
    neutral_items = os.listdir(NEUTRAL_DIR)
    masked_items = os.listdir(MASKED_DIR)
    sunglasses_items = os.listdir(SUNGLASSES_DIR)

    for item in neutral_items:
        if not os.listdir(f"{NEUTRAL_DIR}/{item}"):
            print(f'Neutral folder ->\t{item} has no images, removing directory')
            os.rmdir(f'{NEUTRAL_DIR}/{item}')
    for item in masked_items:
        if not os.listdir(f"{MASKED_DIR}/{item}"):
            print(f'Masked folder ->\t{item} has no images, removing directory')
            os.rmdir(f'{MASKED_DIR}/{item}')
    for item in sunglasses_items:
        if not os.listdir(f"{SUNGLASSES_DIR}/{item}"):
            print(f'Sunglasses folder ->\t{item} has no images, removing directory')
            os.rmdir(f'{SUNGLASSES_DIR}/{item}')


def get_unique_names():
    neutral_items = os.listdir(NEUTRAL_DIR)
    masked_items = list(
        map(lambda x: x.split('_wearing_mask')[0], os.listdir(MASKED_DIR))
    )
    sunglasses_items = list(
        map(lambda x: x.split('_wearing_sunglasses')[0], os.listdir(SUNGLASSES_DIR))
    )
    return sorted(set(masked_items + neutral_items + sunglasses_items))


def get_histogram():
    data = {key: {'Neutral': None, 'Masked': None, 'Sunglasses': None} for key in UNIQUE_NAMES}

    neutral_folders = os.listdir(NEUTRAL_DIR)
    masked_folders = os.listdir(MASKED_DIR)
    sunglasses_folders = os.listdir(SUNGLASSES_DIR)

    for folder in neutral_folders:
        name = folder
        data[name]['Neutral'] = len(os.listdir(f"{NEUTRAL_DIR}/{folder}"))
    for folder in masked_folders:
        name = folder.split('_wearing_mask')[0]
        data[name]['Masked'] = len(os.listdir(f"{MASKED_DIR}/{folder}"))
    for folder in sunglasses_folders:
        name = folder.split('_wearing_sunglasses')[0]
        data[name]['Sunglasses'] = len(os.listdir(f"{SUNGLASSES_DIR}/{folder}"))

    # Remove any person which does not have images for all categories
    data_aux = data.copy()
    for name in data.keys():
        if any(val is None for val in data[name].values()):
            data_aux.pop(name)

            # Also remove directories
            try:
                os.system(f'rm -r {NEUTRAL_DIR}/{name}')
            except FileNotFoundError: pass
            try:
                os.system(f'rm -r {MASKED_DIR}/{name}_wearing_mask')
            except FileNotFoundError: pass
            try:
                os.system(f'rm -r {SUNGLASSES_DIR}/{name}_wearing_sunglasses')
            except FileNotFoundError: pass
    data = data_aux.copy()
    return data


def get_data():
    data = {key: {'Neutral': {}, 'Masked': {}, 'Sunglasses': {}} for key in UNIQUE_NAMES}

    neutral_folders = os.listdir(NEUTRAL_DIR)
    masked_folders = os.listdir(MASKED_DIR)
    sunglasses_folders = os.listdir(SUNGLASSES_DIR)

    for folder in neutral_folders:
        for file in sorted(os.listdir(f"{NEUTRAL_DIR}/{folder}")):
            person_name = folder
            file_name = file.split('.jpg')[0]
            file_path = os.path.join(f"{NEUTRAL_DIR}/{folder}/{file}")
            data[person_name]['Neutral'][file_name] = {
                'file': file_path,
                'embeddings': {
                    'vit': None,
                    'resnet': None,
                    'vgg': None,
                    'inception': None,
                    'mobilenet': None,
                    'efficientnet': None,
                }
            }
    for folder in masked_folders:
        for file in sorted(os.listdir(f"{MASKED_DIR}/{folder}")):
            person_name = folder.split('_wearing_mask')[0]
            file_name = file.split('.jpg')[0]
            file_path = os.path.join(f"{MASKED_DIR}/{folder}/{file}")
            data[person_name]['Masked'][file_name] = {
                'file': file_path,
                'embeddings': {
                    'vit': None,
                    'resnet': None,
                    'vgg': None,
                    'inception': None,
                    'mobilenet': None,
                    'efficientnet': None,
                }
            }
    for folder in sunglasses_folders:
        for file in sorted(os.listdir(f"{SUNGLASSES_DIR}/{folder}")):
            person_name = folder.split('_wearing_sunglasses')[0]
            file_name = file.split('.jpg')[0]
            file_path = os.path.join(f"{SUNGLASSES_DIR}/{folder}/{file}")
            data[person_name]['Sunglasses'][file_name] = {
                'file': file_path,
                'embeddings': {
                    'vit': None,
                    'resnet': None,
                    'vgg': None,
                    'inception': None,
                    'mobilenet': None,
                    'efficientnet': None,
                }
            }
    return data


"""
CREATE DATASET
"""


BASE_DIR = '/mnt/Data/mrt/RealWorldOccludedFaces/images'
NEUTRAL_DIR = f"{BASE_DIR}/neutral"
MASKED_DIR = f"{BASE_DIR}/masked"
SUNGLASSES_DIR = f"{BASE_DIR}/sunglasses"

remove_empty_directories()

UNIQUE_NAMES = get_unique_names()
DATA_HISTOGRAM = get_histogram()
DATA = get_data()
