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
    masked_items = list(map(lambda x: x.split('_wearing_mask')[0], os.listdir(MASKED_DIR)))
    sunglasses_items = list(map(lambda x: x.split('_wearing_sunglasses')[0], os.listdir(SUNGLASSES_DIR)))
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


def compute_score(embeddings1, embeddings2):
    cosine_distance = cosine(embeddings1, embeddings2)
    score = 1 - cosine_distance
    return score


def plot_and_csv(models, ground_truth_, categories, positive_label=1):
    results_dir = os.path.join('./saved_results/Tests/RealWorldOccludedFaces', f"CATEGORIES-{'_'.join(categories)}")
    try:
        os.mkdir(results_dir)
    except FileExistsError:
        pass

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    for model in models.keys():
        model_name_ = models[model]['name']
        model_color_ = models[model]['color']
        model_scores_ = models[model]['scores']

        # Data
        fpr, tpr, thresholds = roc_curve(ground_truth_, model_scores_, pos_label=positive_label)
        auc_result = auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.argmin(np.absolute(fnr - fpr))]
        eer_threshold = thresholds[np.argmin(np.absolute(fnr - fpr))]

        # Plot
        ax.plot(fpr, tpr, linestyle='-', lw=3, color=model_color_, label=f'{model_name_} (EER={eer:.2f}, AUC={auc_result:.3f})')
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


def compute_roc(scores_dict, categories):
    vit_scores = []
    resnet_scores = []
    vgg_scores = []
    inception_scores = []
    mobilenet_scores = []
    efficientnet_scores = []
    ground_truth = []

    for neutral_person_ in scores_dict.keys():
        for neutral_file_ in scores_dict[neutral_person_].keys():
            for probe_person_, values in scores_dict[neutral_person_][neutral_file_].items():

                for category_ in categories:
                    if category_ in probe_person_:
                        vit_scores.append(values['scores']['vit'])
                        resnet_scores.append(values['scores']['resnet'])
                        vgg_scores.append(values['scores']['vgg'])
                        inception_scores.append(values['scores']['inception'])
                        mobilenet_scores.append(values['scores']['mobilenet'])
                        efficientnet_scores.append(values['scores']['efficientnet'])

                        ground_truth.append(1 if neutral_person_ == values['person'] else 0)  # 1 if same person, 0 if different

    models_scores = {
        'vit':          {'name': 'ViT_B32',         'color': 'blue',    'scores': vit_scores},
        'resnet':       {'name': 'ResNet_50',       'color': 'orange',  'scores': resnet_scores},
        'vgg':          {'name': 'VGG_16',          'color': 'green',   'scores': vgg_scores},
        'inception':    {'name': 'Inception_V3',    'color': 'cyan',    'scores': inception_scores},
        'mobilenet':    {'name': 'MobileNet_V2',    'color': 'magenta', 'scores': mobilenet_scores},
        'efficientnet': {'name': 'EfficientNet_B0', 'color': 'brown',   'scores': efficientnet_scores},
    }
    plot_and_csv(models_scores, ground_truth, categories)


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


"""
MATCH IMAGES TO OBTAIN MATCHING SCORES
"""


try:
    with open('./saved_results/Tests/RealWorldOccludedFaces/scores.pickle', 'rb') as scores_file:
        SCORES = pickle.load(scores_file)
except FileNotFoundError:
    SCORES = {person: {} for person in DATA.keys()}
    compare_with = [
        'masked',
        'sunglasses',
    ]

    for neutral_person in SCORES.keys():
        for neutral_file in sorted(os.listdir(f'{NEUTRAL_DIR}/{neutral_person}')):
            neutral_id = f"{neutral_person}_{neutral_file.split('.jpg')[0]}"
            SCORES[neutral_person][neutral_id] = {}

            if 'masked' in compare_with:
                for masked_person in sorted(os.listdir(MASKED_DIR)):
                    for masked_file in sorted(os.listdir(f'{MASKED_DIR}/{masked_person}')):
                        masked_id = f"{masked_person}_{masked_file.split('.jpg')[0]}"

                        SCORES[neutral_person][neutral_id][masked_id] = {
                            'person': masked_person.split('_wearing_mask')[0],
                            'scores': {
                                'vit': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['vit'],
                                    DATA[masked_person.split('_wearing_mask')[0]]['Masked'][masked_file.split('.jpg')[0]]['embeddings']['vit']
                                ),
                                'resnet': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['resnet'],
                                    DATA[masked_person.split('_wearing_mask')[0]]['Masked'][masked_file.split('.jpg')[0]]['embeddings']['resnet']
                                ),
                                'vgg': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['vgg'],
                                    DATA[masked_person.split('_wearing_mask')[0]]['Masked'][masked_file.split('.jpg')[0]]['embeddings']['vgg']
                                ),
                                'inception': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['inception'],
                                    DATA[masked_person.split('_wearing_mask')[0]]['Masked'][masked_file.split('.jpg')[0]]['embeddings']['inception']
                                ),
                                'mobilenet': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['mobilenet'],
                                    DATA[masked_person.split('_wearing_mask')[0]]['Masked'][masked_file.split('.jpg')[0]]['embeddings']['mobilenet']
                                ),
                                'efficientnet': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['efficientnet'],
                                    DATA[masked_person.split('_wearing_mask')[0]]['Masked'][masked_file.split('.jpg')[0]]['embeddings']['efficientnet']
                                ),
                            }
                        }

            if 'sunglasses' in compare_with:
                for sunglasses_person in sorted(os.listdir(SUNGLASSES_DIR)):
                    for sunglasses_file in sorted(os.listdir(f'{SUNGLASSES_DIR}/{sunglasses_person}')):
                        sunglasses_id = f"{sunglasses_person}_{sunglasses_file.split('.jpg')[0]}"

                        SCORES[neutral_person][neutral_id][sunglasses_id] = {
                            'person': sunglasses_person.split('_wearing_sunglasses')[0],
                            'scores': {
                                'vit': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['vit'],
                                    DATA[sunglasses_person.split('_wearing_sunglasses')[0]]['Sunglasses'][sunglasses_file.split('.jpg')[0]]['embeddings']['vit']
                                ),
                                'resnet': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['resnet'],
                                    DATA[sunglasses_person.split('_wearing_sunglasses')[0]]['Sunglasses'][sunglasses_file.split('.jpg')[0]]['embeddings']['resnet']
                                ),
                                'vgg': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['vgg'],
                                    DATA[sunglasses_person.split('_wearing_sunglasses')[0]]['Sunglasses'][sunglasses_file.split('.jpg')[0]]['embeddings']['vgg']
                                ),
                                'inception': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['inception'],
                                    DATA[sunglasses_person.split('_wearing_sunglasses')[0]]['Sunglasses'][sunglasses_file.split('.jpg')[0]]['embeddings']['inception']
                                ),
                                'mobilenet': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['mobilenet'],
                                    DATA[sunglasses_person.split('_wearing_sunglasses')[0]]['Sunglasses'][sunglasses_file.split('.jpg')[0]]['embeddings']['mobilenet']
                                ),
                                'efficientnet': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['efficientnet'],
                                    DATA[sunglasses_person.split('_wearing_sunglasses')[0]]['Sunglasses'][sunglasses_file.split('.jpg')[0]]['embeddings']['efficientnet']
                                ),
                            }
                        }

    with open('./saved_results/Tests/RealWorldOccludedFaces/scores.pickle', 'wb') as scores_file:
        pickle.dump(SCORES, scores_file)


"""
COMPUTE ROC CURVES
"""


compute_roc(
    SCORES,
    categories=[
        'mask',
        'sunglasses',
    ]
)
