import os
import matplotlib.pyplot as plt


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

    neutral_items = os.listdir(NEUTRAL_DIR)
    masked_items = os.listdir(MASKED_DIR)
    sunglasses_items = os.listdir(SUNGLASSES_DIR)

    for item in neutral_items:
        name = item
        data[name]['Neutral'] = len(os.listdir(f"{NEUTRAL_DIR}/{item}"))
    for item in masked_items:
        name = item.split('_wearing_mask')[0]
        data[name]['Masked'] = len(os.listdir(f"{MASKED_DIR}/{item}"))
    for item in sunglasses_items:
        name = item.split('_wearing_sunglasses')[0]
        data[name]['Sunglasses'] = len(os.listdir(f"{SUNGLASSES_DIR}/{item}"))
    return data


BASE_DIR = '/mnt/Data/mrt/RealWorldOccludedFaces/images'
NEUTRAL_DIR = f"{BASE_DIR}/neutral"
MASKED_DIR = f"{BASE_DIR}/masked"
SUNGLASSES_DIR = f"{BASE_DIR}/sunglasses"

remove_empty_directories()

UNIQUE_NAMES = get_unique_names()

DATA_HISTOGRAM = get_histogram()
