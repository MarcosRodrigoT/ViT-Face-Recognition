import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from vit_keras import vit
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc


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


def preprocess_image(img_path):
    img_ = tf.io.read_file(img_path)
    img_ = tf.image.decode_jpeg(img_, channels=3)
    img_ = tf.image.convert_image_dtype(img_, dtype=tf.float32)
    img_ = tf.image.resize(img_, [224, 224])
    img_ = tf.expand_dims(img_, axis=0)
    return img_


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


"""
LOAD MODELS
"""


IMAGE_SIZE = 224
NUM_CLASSES = 8631

""" ViT_B32 """
vit_model = vit.vit_b32(
    image_size=IMAGE_SIZE,
    pretrained=True,
    include_top=False,
    pretrained_top=False,
)
y = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(vit_model.output)
vit_model = tf.keras.models.Model(inputs=vit_model.input, outputs=y)

vit_model.load_weights("./saved_results/Models/ViT_B32/checkpoint").expect_partial()   # suppresses warnings
vit_model = tf.keras.models.Model(inputs=vit_model.input, outputs=vit_model.layers[-2].output)
vit_model.summary()

""" ResNet_50 """
resnet50_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(resnet50_model.output)
Y = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
resnet50_model = tf.keras.models.Model(inputs=resnet50_model.input, outputs=Y, name='ResNet50')

resnet50_model.load_weights("./saved_results/Models/ResNet_50/checkpoint").expect_partial()   # suppresses warnings
resnet50_model = tf.keras.models.Model(inputs=resnet50_model.input, outputs=resnet50_model.layers[-2].output)
resnet50_model.summary()

""" VGG_16 """
vgg16_model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling=None,
)
Y = vgg16_model.layers[-2].output
Y = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform)(Y)
vgg16_model = tf.keras.models.Model(inputs=vgg16_model.input, outputs=Y, name='VGG16')

vgg16_model.load_weights("./saved_results/Models/VGG_16/checkpoint").expect_partial()   # suppresses warnings
vgg16_model = tf.keras.models.Model(inputs=vgg16_model.input, outputs=vgg16_model.layers[-2].output)
vgg16_model.summary()

""" Inception_v3 """
inception_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(inception_model.output)
Y = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
inception_model = tf.keras.models.Model(inputs=inception_model.input, outputs=Y, name='InceptionV3')
inception_model.summary()

inception_model.load_weights("./saved_results/Models/Inception_V3/checkpoint").expect_partial()   # suppresses warnings
inception_model = tf.keras.models.Model(inputs=inception_model.input, outputs=inception_model.layers[-2].output)
inception_model.summary()

""" MobileNet_v2 """
mobilenet_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(mobilenet_model.output)
Y = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
mobilenet_model = tf.keras.models.Model(inputs=mobilenet_model.input, outputs=Y, name='MobileNetV2')
mobilenet_model.summary()

mobilenet_model.load_weights("./saved_results/Models/MobileNet_V2/checkpoint").expect_partial()   # suppresses warnings
mobilenet_model = tf.keras.models.Model(inputs=mobilenet_model.input, outputs=mobilenet_model.layers[-2].output)
mobilenet_model.summary()

""" EfficientNet_B0 """
efficientnetB0_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(efficientnetB0_model.output)
Y = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
efficientnetB0_model = tf.keras.models.Model(inputs=efficientnetB0_model.input, outputs=Y, name='EfficientNetB0')
efficientnetB0_model.summary()

efficientnetB0_model.load_weights("./saved_results/Models/EfficientNet_B0/checkpoint").expect_partial()   # suppresses warnings
efficientnetB0_model = tf.keras.models.Model(inputs=efficientnetB0_model.input, outputs=efficientnetB0_model.layers[-2].output)
efficientnetB0_model.summary()


"""
PREPROCESS IMAGES AND COMPUTE EMBEDDINGS
"""


try:
    with open('./saved_results/Tests/RealWorldOccludedFaces/embeddings.pickle', 'rb') as embeddings_file:
        DATA = pickle.load(embeddings_file)
except FileNotFoundError:
    for person in DATA.keys():
        for category in DATA[person].keys():
            for element in DATA[person][category].keys():
                img = preprocess_image(DATA[person][category][element]['file'])

                embeddings_vit = vit_model(img).numpy()
                embeddings_resnet = resnet50_model(img).numpy()
                embeddings_vgg16 = vgg16_model(img).numpy()
                embeddings_inception = inception_model(img).numpy()
                embeddings_mobilenet = mobilenet_model(img).numpy()
                embeddings_efficientnet = efficientnetB0_model(img).numpy()

                DATA[person][category][element]['embeddings']['vit'] = embeddings_vit
                DATA[person][category][element]['embeddings']['resnet'] = embeddings_resnet
                DATA[person][category][element]['embeddings']['vgg'] = embeddings_vgg16
                DATA[person][category][element]['embeddings']['inception'] = embeddings_inception
                DATA[person][category][element]['embeddings']['mobilenet'] = embeddings_mobilenet
                DATA[person][category][element]['embeddings']['efficientnet'] = embeddings_efficientnet

    with open('./saved_results/Tests/RealWorldOccludedFaces/embeddings.pickle', 'wb') as embeddings_file:
        pickle.dump(DATA, embeddings_file)
