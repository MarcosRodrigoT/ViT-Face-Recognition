import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from vit_keras import vit
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
from tqdm import tqdm


def preprocess_image(img_path):
    img_ = tf.io.read_file(img_path)
    img_ = tf.image.decode_jpeg(img_, channels=3)
    img_ = tf.image.convert_image_dtype(img_, dtype=tf.float32)
    img_ = tf.image.resize(img_, [224, 224])
    img_ = tf.expand_dims(img_, axis=0)
    return img_


def compute_score(embeddings1, embeddings2):
    cosine_distance = cosine(embeddings1, embeddings2)
    score = 1 - cosine_distance
    return score


def plot_and_csv(models, ground_truth_, cameras, distances, positive_label=1):
    results_dir = os.path.join('./saved_results/Tests/SCface', f"ROC-CAMERAS-{'_'.join(cameras)}-DISTANCES-{'_'.join(distances)}")
    try:
        os.mkdir(results_dir)
    except FileExistsError:
        pass

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    for model in tqdm(models.keys(), desc="Processing model"):
        model_name_ = models[model]['name']
        model_color_ = models[model]['color']
        model_scores_ = models[model]['scores']

        # Data
        fpr, tpr, thresholds = roc_curve(ground_truth_, model_scores_, pos_label=positive_label)
        auc_result = auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.argmin(np.absolute(fnr - fpr))]
        eer_threshold = thresholds[np.argmin(np.absolute(fnr - fpr))]

        # Find the maximum F1 score and corresponding threshold
        max_f1 = 0
        max_f1_recall = 0
        max_f1_precision = 0

        for thresh in tqdm(thresholds, desc="Processing thresholds"):
            binarized_results = [1 if score >= thresh else 0 for score in model_scores_]
            current_fscore = f1_score(ground_truth_, binarized_results)
            if current_fscore > max_f1:
                max_f1 = current_fscore
                max_f1_recall = recall_score(ground_truth_, binarized_results)
                max_f1_precision = precision_score(ground_truth_, binarized_results)

        # Plot
        ax.plot(fpr, tpr, linestyle='-', lw=3, color=model_color_, label=f'{model_name_} (EER={eer:.2f}, AUC={auc_result:.3f}, R={max_f1_recall:.3f}, P={max_f1_precision:.3f}, F={max_f1:.3f})')
        ax.scatter(eer, tpr[np.argmin(np.absolute(fnr - fpr))], color=model_color_, linewidths=8, zorder=10)

        # CSV
        result_pd = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        result_pd['EER'] = pd.DataFrame([eer, tpr[np.argmin(np.absolute(fnr - fpr))]])
        result_pd.to_csv(f"{results_dir}/{model_name_}_ROC.csv", header=True, index=False)

    ax.set_title('Receiver Operating Characteristics (ROC)', fontsize=15)
    ax.set_xlabel('FPR', fontsize=15)
    ax.set_ylabel('TPR', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(loc='lower right', prop={"size": 11})

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.savefig(f"{results_dir}/ROC.png", bbox_inches='tight')

    ax.set_xlim([0.0, 0.3])
    ax.set_ylim([0.7, 1.0])
    plt.savefig(f"{results_dir}/ROC_zoom.png", bbox_inches='tight')


def compute_roc(scores_dict, cameras, distances):
    vit_scores = []
    resnet_scores = []
    vgg_scores = []
    inception_scores = []
    mobilenet_scores = []
    efficientnet_scores = []
    ground_truth = []

    for mug_person_ in scores_dict.keys():
        for sur_item_, sur_values_ in scores_dict[mug_person_].items():
            person_ = sur_values_['person']
            cam_ = sur_values_['camera']
            dist_ = sur_values_['distance']

            if cam_ in cameras and dist_ in distances:
                vit_scores.append(scores_dict[mug_person_][sur_item_]['vit'])
                resnet_scores.append(scores_dict[mug_person_][sur_item_]['resnet'])
                vgg_scores.append(scores_dict[mug_person_][sur_item_]['vgg'])
                inception_scores.append(scores_dict[mug_person_][sur_item_]['inception'])
                mobilenet_scores.append(scores_dict[mug_person_][sur_item_]['mobilenet'])
                efficientnet_scores.append(scores_dict[mug_person_][sur_item_]['efficientnet'])

                ground_truth.append(1 if person_ == mug_person_ else 0)  # 1 if same person, 0 if different

    models_scores = {
        'vit':          {'name': 'ViT_B32',         'color': 'blue',    'scores': vit_scores},
        'resnet':       {'name': 'ResNet_50',       'color': 'orange',  'scores': resnet_scores},
        'vgg':          {'name': 'VGG_16',          'color': 'green',   'scores': vgg_scores},
        'inception':    {'name': 'Inception_V3',    'color': 'cyan',    'scores': inception_scores},
        'mobilenet':    {'name': 'MobileNet_V2',    'color': 'magenta', 'scores': mobilenet_scores},
        'efficientnet': {'name': 'EfficientNet_B0', 'color': 'brown',   'scores': efficientnet_scores},
    }
    plot_and_csv(models_scores, ground_truth, cameras, distances)


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
        embeddings_resnet = resnet50_model(img).numpy()
        embeddings_vgg16 = vgg16_model(img).numpy()
        embeddings_inception = inception_model(img).numpy()
        embeddings_mobilenet = mobilenet_model(img).numpy()
        embeddings_efficientnet = efficientnetB0_model(img).numpy()

        mugshot_data[person]['embeddings']['vit'] = embeddings_vit
        mugshot_data[person]['embeddings']['resnet'] = embeddings_resnet
        mugshot_data[person]['embeddings']['vgg'] = embeddings_vgg16
        mugshot_data[person]['embeddings']['inception'] = embeddings_inception
        mugshot_data[person]['embeddings']['mobilenet'] = embeddings_mobilenet
        mugshot_data[person]['embeddings']['efficientnet'] = embeddings_efficientnet

    for person in surveillance_data.keys():
        for file in surveillance_data[person].keys():
            img = preprocess_image(surveillance_data[person][file]['file'])

            embeddings_vit = vit_model(img).numpy()
            embeddings_resnet = resnet50_model(img).numpy()
            embeddings_vgg16 = vgg16_model(img).numpy()
            embeddings_inception = inception_model(img).numpy()
            embeddings_mobilenet = mobilenet_model(img).numpy()
            embeddings_efficientnet = efficientnetB0_model(img).numpy()

            surveillance_data[person][file]['embeddings']['vit'] = embeddings_vit
            surveillance_data[person][file]['embeddings']['resnet'] = embeddings_resnet
            surveillance_data[person][file]['embeddings']['vgg'] = embeddings_vgg16
            surveillance_data[person][file]['embeddings']['inception'] = embeddings_inception
            surveillance_data[person][file]['embeddings']['mobilenet'] = embeddings_mobilenet
            surveillance_data[person][file]['embeddings']['efficientnet'] = embeddings_efficientnet

    with open('./saved_results/Tests/SCface/embeddings.pickle', 'wb') as file:
        data = (mugshot_data, surveillance_data)
        pickle.dump(data, file)


"""
MATCH MUGSHOT AND SURVEILLANCE IMAGES TO OBTAIN MATCHING SCORES
"""


try:
    with open('./saved_results/Tests/SCface/scores.pickle', 'rb') as scores_file:
        scores = pickle.load(scores_file)
except FileNotFoundError:
    scores = {person: {} for person in mugshot_data.keys()}
    for mug_person in mugshot_data.keys():
        for sur_person in surveillance_data.keys():
            for file in surveillance_data[sur_person].keys():
                scores[mug_person][file.split('.jpg')[0]] = {
                    'person': file.split('_')[0],
                    'camera': surveillance_data[sur_person][file]['camera'],
                    'distance': surveillance_data[sur_person][file]['distance'],
                    'vit': compute_score(
                        mugshot_data[mug_person]['embeddings']['vit'],
                        surveillance_data[sur_person][file]['embeddings']['vit']
                    ),
                    'resnet': compute_score(
                        mugshot_data[mug_person]['embeddings']['resnet'],
                        surveillance_data[sur_person][file]['embeddings']['resnet']
                    ),
                    'vgg': compute_score(
                        mugshot_data[mug_person]['embeddings']['vgg'],
                        surveillance_data[sur_person][file]['embeddings']['vgg']
                    ),
                    'inception': compute_score(
                        mugshot_data[mug_person]['embeddings']['inception'],
                        surveillance_data[sur_person][file]['embeddings']['inception']
                    ),
                    'mobilenet': compute_score(
                        mugshot_data[mug_person]['embeddings']['mobilenet'],
                        surveillance_data[sur_person][file]['embeddings']['mobilenet']
                    ),
                    'efficientnet': compute_score(
                        mugshot_data[mug_person]['embeddings']['efficientnet'],
                        surveillance_data[sur_person][file]['embeddings']['efficientnet']
                    ),
                }

    with open('./saved_results/Tests/SCface/scores.pickle', 'wb') as scores_file:
        pickle.dump(scores, scores_file)


"""
COMPUTE ROC CURVES
"""


compute_roc(
    scores,
    cameras=[
        'cam1',
        'cam2',
        'cam3',
        'cam4',
        'cam5',
        # 'cam6',
        # 'cam7',
        # 'cam8',
    ],
    distances=[
        '1',
        '2',
        '3',
        # 'None',
    ]
)
