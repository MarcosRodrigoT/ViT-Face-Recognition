import os
import pickle
import tensorflow as tf
from vit_keras import vit


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
BASE_DIR = '/mnt/Data/mrt/SCface_database'
MUGSHOT_DIR = f'{BASE_DIR}/mugshot_frontal_cropped_all'
SURVEILLANCE_DIR = f'{BASE_DIR}/surveillance_cameras_all'

mugshot_data = {}
for file in sorted(os.listdir(MUGSHOT_DIR)):
    person = file.split('_')[0]
    file_path = os.path.join(MUGSHOT_DIR, file)
    mugshot_data[person] = {
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

surveillance_data = {person: {} for person in mugshot_data.keys()}
for file in sorted(os.listdir(SURVEILLANCE_DIR)):
    components = file.split('.')[0].split('_')
    if len(components) == 3:
        person, camera, distance = components
    else:
        person, camera, distance = components + ['None']
    file_path = os.path.join(SURVEILLANCE_DIR, file)

    surveillance_data[person][file] = {
        'file': file_path,
        'camera': camera,
        'distance': distance,
        'embeddings': {
            'vit': None,
            'resnet': None,
            'vgg': None,
            'inception': None,
            'mobilenet': None,
            'efficientnet': None,
        }
    }


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
    with open('./saved_results/Tests/SCface/embeddings.pickle', 'rb') as file:
        mugshot_data, surveillance_data = pickle.load(file)
except FileNotFoundError:
    for person in mugshot_data.keys():
        img = preprocess_image(mugshot_data[person]['file'])

        embeddings_vit = vit_model(img).numpy()
        embeddings1_resnet = resnet50_model(img).numpy()
        embeddings1_vgg16 = vgg16_model(img).numpy()
        embeddings1_inception = inception_model(img).numpy()
        embeddings1_mobilenet = mobilenet_model(img).numpy()
        embeddings1_efficientnet = efficientnetB0_model(img).numpy()

        mugshot_data[person]['embeddings']['vit'] = embeddings_vit
        mugshot_data[person]['embeddings']['resnet'] = embeddings1_resnet
        mugshot_data[person]['embeddings']['vgg'] = embeddings1_vgg16
        mugshot_data[person]['embeddings']['inception'] = embeddings1_inception
        mugshot_data[person]['embeddings']['mobilenet'] = embeddings1_mobilenet
        mugshot_data[person]['embeddings']['efficientnet'] = embeddings1_efficientnet

    for person in surveillance_data.keys():
        for file in surveillance_data[person].keys():
            img = preprocess_image(surveillance_data[person][file]['file'])

            embeddings_vit = vit_model(img).numpy()
            embeddings1_resnet = resnet50_model(img).numpy()
            embeddings1_vgg16 = vgg16_model(img).numpy()
            embeddings1_inception = inception_model(img).numpy()
            embeddings1_mobilenet = mobilenet_model(img).numpy()
            embeddings1_efficientnet = efficientnetB0_model(img).numpy()

            surveillance_data[person][file]['embeddings']['vit'] = embeddings_vit
            surveillance_data[person][file]['embeddings']['resnet'] = embeddings1_resnet
            surveillance_data[person][file]['embeddings']['vgg'] = embeddings1_vgg16
            surveillance_data[person][file]['embeddings']['inception'] = embeddings1_inception
            surveillance_data[person][file]['embeddings']['mobilenet'] = embeddings1_mobilenet
            surveillance_data[person][file]['embeddings']['efficientnet'] = embeddings1_efficientnet

    with open('./saved_results/Tests/SCface/embeddings.pickle', 'wb') as file:
        data = (mugshot_data, surveillance_data)
        pickle.dump(data, file)
