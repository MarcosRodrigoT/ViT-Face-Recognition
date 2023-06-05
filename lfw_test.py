import tensorflow as tf
from vit_keras import vit
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def preprocess(paths):
    path_img1, path_img2 = paths

    img1 = tf.io.read_file(path_img1)
    img1 = tf.image.decode_jpeg(img1, channels=3)
    img1 = tf.image.convert_image_dtype(img1, dtype=tf.float32)
    img1 = tf.image.resize(img1, [224, 224])
    img1 = tf.expand_dims(img1, axis=0)

    img2 = tf.io.read_file(path_img2)
    img2 = tf.image.decode_jpeg(img2, channels=3)
    img2 = tf.image.convert_image_dtype(img2, dtype=tf.float32)
    img2 = tf.image.resize(img2, [224, 224])
    img2 = tf.expand_dims(img2, axis=0)

    return [img1, img2]


def compute_score(embeddings1, embeddings2):
    cosine_distance = cosine(embeddings1, embeddings2)
    score = 1 - cosine_distance

    return score


def compute_roc(results_dictionary, fig_name, positive_label=1):
    vit_results = []
    resnet_results = []
    vgg_results = []
    inception_results = []
    mobilenet_results = []
    efficientnet_results = []
    gt_results = []

    for pair_example in results_dictionary.keys():
        vit_results.append(results_dictionary[pair_example]['vit'])
        resnet_results.append(results_dictionary[pair_example]['resnet'])
        vgg_results.append(results_dictionary[pair_example]['vgg'])
        inception_results.append(results_dictionary[pair_example]['inception'])
        mobilenet_results.append(results_dictionary[pair_example]['mobilenet'])
        efficientnet_results.append(results_dictionary[pair_example]['efficientnet'])
        gt_results.append(results_dictionary[pair_example]['GT'])

    # Figures
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    lw = 3

    """ ViT_B32 """
    # Data
    fpr_vit, tpr_vit, thresholds_vit = roc_curve(gt_results, vit_results, pos_label=positive_label)
    auc_vit = auc(fpr_vit, tpr_vit)
    fnr_vit = 1 - tpr_vit
    eer_vit = fpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]
    eer_vit_threshold = thresholds_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]

    # Plot
    ax.plot(fpr_vit, tpr_vit, linestyle='-', lw=lw, color='blue', label=f'ViT_B32 (EER={eer_vit:.2f}, AUC={auc_vit:.3f})')
    ax.scatter(eer_vit, tpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))], color='blue', linewidths=8, zorder=10)

    # CSV
    vit_pd = pd.DataFrame({'FPR_ViT': fpr_vit, 'TPR_ViT': tpr_vit})
    vit_pd['EER_ViT'] = pd.DataFrame([eer_vit, tpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]])
    vit_pd.to_csv('./saved_results/Tests/LFW/ViT_B32_ROC.csv', header=True, index=False)

    """ ResNet_50 """
    # Data
    fpr_resnet, tpr_resnet, thresholds_resnet = roc_curve(gt_results, resnet_results, pos_label=positive_label)
    auc_resnet = auc(fpr_resnet, tpr_resnet)
    fnr_resnet = 1 - tpr_resnet
    eer_resnet = fpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]
    eer_resnet_threshold = thresholds_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]

    # Plot
    ax.plot(fpr_resnet, tpr_resnet, linestyle='-', lw=lw, color='orange', label=f'ResNet_50 (EER={eer_resnet:.2f}, AUC={auc_resnet:.3f})')
    ax.scatter(eer_resnet, tpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))], color='orange', linewidths=8, zorder=10)

    # CSV
    resnet_pd = pd.DataFrame({'FPR_RESNET': fpr_resnet, 'TPR_RESNET': tpr_resnet})
    resnet_pd['EER_RESNET'] = pd.DataFrame([eer_resnet, tpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]])
    resnet_pd.to_csv('./saved_results/Tests/LFW/ResNet_50_ROC.csv', header=True, index=False)

    """ VGG_16 """
    # Data
    fpr_vgg, tpr_vgg, thresholds_vgg = roc_curve(gt_results, vgg_results, pos_label=positive_label)
    auc_vgg = auc(fpr_vgg, tpr_vgg)
    fnr_vgg = 1 - tpr_vgg
    eer_vgg = fpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]
    eer_vgg_threshold = thresholds_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]

    # Plot
    ax.plot(fpr_vgg, tpr_vgg, linestyle='-', lw=lw, color='green', label=f'VGG_16 (EER={eer_vgg:.2f}, AUC={auc_vgg:.3f})')
    ax.scatter(eer_vgg, tpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))], color='green', linewidths=8, zorder=10)

    # CSV
    vgg_pd = pd.DataFrame({'FPR_VGG': fpr_vgg, 'TPR_VGG': tpr_vgg})
    vgg_pd['EER_VGG'] = pd.DataFrame([eer_vgg, tpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]])
    vgg_pd.to_csv('./saved_results/Tests/LFW/VGG_16_ROC.csv', header=True, index=False)

    """ Inception_V3 """
    # Data
    fpr_inception, tpr_inception, thresholds_inception = roc_curve(gt_results, inception_results, pos_label=positive_label)
    auc_inception = auc(fpr_inception, tpr_inception)
    fnr_inception = 1 - tpr_inception
    eer_inception = fpr_inception[np.argmin(np.absolute(fnr_inception - fpr_inception))]
    eer_inception_threshold = thresholds_inception[np.argmin(np.absolute(fnr_inception - fpr_inception))]

    # Plot
    ax.plot(fpr_inception, tpr_inception, linestyle='-', lw=lw, color='cyan', label=f'Inception_V3 (EER={eer_inception:.2f}, AUC={auc_inception:.3f})')
    ax.scatter(eer_inception, tpr_inception[np.argmin(np.absolute(fnr_inception - fpr_inception))], color='cyan', linewidths=8, zorder=10)

    # CSV
    inception_pd = pd.DataFrame({'FPR_INCEPTION': fpr_inception, 'TPR_INCEPTION': tpr_inception})
    inception_pd['EER_INCEPTION'] = pd.DataFrame([eer_inception, tpr_inception[np.argmin(np.absolute(fnr_inception - fpr_inception))]])
    inception_pd.to_csv('./saved_results/Tests/LFW/Inception_V3_ROC.csv', header=True, index=False)

    """ MobileNet_V2 """
    # Data
    fpr_mobilenet, tpr_mobilenet, thresholds_mobilenet = roc_curve(gt_results, mobilenet_results, pos_label=positive_label)
    auc_mobilenet = auc(fpr_mobilenet, tpr_mobilenet)
    fnr_mobilenet = 1 - tpr_mobilenet
    eer_mobilenet = fpr_mobilenet[np.argmin(np.absolute(fnr_mobilenet - fpr_mobilenet))]
    eer_mobilenet_threshold = thresholds_mobilenet[np.argmin(np.absolute(fnr_mobilenet - fpr_mobilenet))]

    # Plot
    ax.plot(fpr_mobilenet, tpr_mobilenet, linestyle='-', lw=lw, color='magenta', label=f'MobileNet_V2 (EER={eer_mobilenet:.2f}, AUC={auc_mobilenet:.3f})')
    ax.scatter(eer_mobilenet, tpr_mobilenet[np.argmin(np.absolute(fnr_mobilenet - fpr_mobilenet))], color='magenta', linewidths=8, zorder=10)

    # CSV
    mobilenet_pd = pd.DataFrame({'FPR_MOBILENET': fpr_mobilenet, 'TPR_MOBILENET': tpr_mobilenet})
    mobilenet_pd['EER_MOBILENET'] = pd.DataFrame([eer_mobilenet, tpr_mobilenet[np.argmin(np.absolute(fnr_mobilenet - fpr_mobilenet))]])
    mobilenet_pd.to_csv('./saved_results/Tests/LFW/MobileNet_V2_ROC.csv', header=True, index=False)

    """ EfficientNet_B0 """
    # Data
    fpr_efficientnet, tpr_efficientnet, thresholds_efficientnet = roc_curve(gt_results, efficientnet_results, pos_label=positive_label)
    auc_efficientnet = auc(fpr_efficientnet, tpr_efficientnet)
    fnr_efficientnet = 1 - tpr_efficientnet
    eer_efficientnet = fpr_efficientnet[np.argmin(np.absolute(fnr_efficientnet - fpr_efficientnet))]
    eer_efficientnet_threshold = thresholds_efficientnet[np.argmin(np.absolute(fnr_efficientnet - fpr_efficientnet))]

    # Plot
    ax.plot(fpr_efficientnet, tpr_efficientnet, linestyle='-', lw=lw, color='brown', label=f'EfficientNet_B0 (EER={eer_efficientnet:.2f}, AUC={auc_efficientnet:.3f})')
    ax.scatter(eer_efficientnet, tpr_efficientnet[np.argmin(np.absolute(fnr_efficientnet - fpr_efficientnet))], color='brown', linewidths=8, zorder=10)

    # CSV
    efficientnet_pd = pd.DataFrame({'FPR_EFFICIENTNET': fpr_efficientnet, 'TPR_EFFICIENTNET': tpr_efficientnet})
    efficientnet_pd['EER_EFFICIENTNET'] = pd.DataFrame([eer_efficientnet, tpr_efficientnet[np.argmin(np.absolute(fnr_efficientnet - fpr_efficientnet))]])
    efficientnet_pd.to_csv('./saved_results/Tests/LFW/EfficientNet_B0_ROC.csv', header=True, index=False)

    ax.set_title('Receiver Operating Characteristics (ROC)', fontsize=15)
    ax.set_xlabel('FPR', fontsize=15)
    ax.set_ylabel('TPR', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(loc='lower right', prop={"size": 11})

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.savefig(f"./saved_results/Tests/LFW/{fig_name}.png", bbox_inches='tight')

    ax.set_xlim([0.0, 0.3])
    ax.set_ylim([0.7, 1.0])
    plt.savefig(f"./saved_results/Tests/LFW/{fig_name}_zoom.png", bbox_inches='tight')


"""
CREATE DATASET WITH PAIRS
"""

pairs = {}
with open('./datasets/LFW/pairs.txt', 'r') as file:
    for idx, line in enumerate(file):
        line_aux = line.split('\n')[0].split('\t')
        if len(line_aux) < 4:
            line_aux.insert(2, line_aux[0])

        img_first = f"{line_aux[0]}_{''.join(str(i) for i in [0 for _ in range(4 - len(line_aux[1]))])}{line_aux[1]}.jpg"
        img_second = f"{line_aux[2]}_{''.join(str(i) for i in [0 for _ in range(4 - len(line_aux[3]))])}{line_aux[3]}.jpg"
        pairs[idx] = [
            f".datasets/LFW/lfw/{line_aux[0]}/{img_first}",
            f".datasets/LFW/lfw/{line_aux[2]}/{img_second}",
            1 if line_aux[0] == line_aux[2] else 0,     # 1 if same person, 0 if different
        ]

print(f"[INFO] Pairs:")
for key, val in pairs.items():
    print(f"Pair {key}: {val[0]} \t->\t {val[1]}")


"""
LOAD MODELS
"""

image_size = 224
num_classes = 8631

""" ViT_B32 """
vit_model = vit.vit_b32(
    image_size=image_size,
    pretrained=True,
    include_top=False,
    pretrained_top=False,
)
y = tf.keras.layers.Dense(num_classes, activation='softmax')(vit_model.output)
vit_model = tf.keras.models.Model(inputs=vit_model.input, outputs=y)

vit_model.load_weights("./saved_results/Models/ViT_B32/checkpoint").expect_partial()   # suppresses warnings
vit_model = tf.keras.models.Model(inputs=vit_model.input, outputs=vit_model.layers[-2].output)
vit_model.summary()

""" ResNet_50 """
resnet50_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(resnet50_model.output)
Y = tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
resnet50_model = tf.keras.models.Model(inputs=resnet50_model.input, outputs=Y, name='ResNet50')

resnet50_model.load_weights("./saved_results/Models/ResNet_50/checkpoint").expect_partial()   # suppresses warnings
resnet50_model = tf.keras.models.Model(inputs=resnet50_model.input, outputs=resnet50_model.layers[-2].output)
resnet50_model.summary()

""" VGG_16 """
vgg16_model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling=None,
)
Y = vgg16_model.layers[-2].output
Y = tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform)(Y)
vgg16_model = tf.keras.models.Model(inputs=vgg16_model.input, outputs=Y, name='VGG16')

vgg16_model.load_weights("./saved_results/Models/VGG_16/checkpoint").expect_partial()   # suppresses warnings
vgg16_model = tf.keras.models.Model(inputs=vgg16_model.input, outputs=vgg16_model.layers[-2].output)
vgg16_model.summary()

""" Inception_v3 """
inception_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(inception_model.output)
Y = tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
inception_model = tf.keras.models.Model(inputs=inception_model.input, outputs=Y, name='InceptionV3')
inception_model.summary()

inception_model.load_weights("./saved_results/Models/Inception_V3/checkpoint").expect_partial()   # suppresses warnings
inception_model = tf.keras.models.Model(inputs=inception_model.input, outputs=inception_model.layers[-2].output)
inception_model.summary()

""" MobileNet_v2 """
mobilenet_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(mobilenet_model.output)
Y = tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
mobilenet_model = tf.keras.models.Model(inputs=mobilenet_model.input, outputs=Y, name='MobileNetV2')
mobilenet_model.summary()

mobilenet_model.load_weights("./saved_results/Models/MobileNet_V2/checkpoint").expect_partial()   # suppresses warnings
mobilenet_model = tf.keras.models.Model(inputs=mobilenet_model.input, outputs=mobilenet_model.layers[-2].output)
mobilenet_model.summary()

""" EfficientNet_B0 """
efficientnetB0_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(efficientnetB0_model.output)
Y = tf.keras.layers.Dense(units=num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
efficientnetB0_model = tf.keras.models.Model(inputs=efficientnetB0_model.input, outputs=Y, name='EfficientNetB0')
efficientnetB0_model.summary()

efficientnetB0_model.load_weights("./saved_results/Models/EfficientNet_B0/checkpoint").expect_partial()   # suppresses warnings
efficientnetB0_model = tf.keras.models.Model(inputs=efficientnetB0_model.input, outputs=efficientnetB0_model.layers[-2].output)
efficientnetB0_model.summary()


"""
PREPROCESS IMAGE PAIRS AND COMPUTE SCORE MODELS
"""

try:
    with open('./saved_results/Tests/LFW/results.pickle', 'rb') as file:
        results = pickle.load(file)
except FileNotFoundError:
    results = {}
    for key, val in pairs.items():
        img1_, img2_ = preprocess(val[:-1])
        print(f"[INFO] Pair {key}")
        print(
            f"Image 1 shape: {img1_.shape}\n"
            f"Image 1 dtype: {img1_.dtype}\n"
            f"Image 1 Min val: {tf.reduce_min(img1_).numpy()}\n"
            f"Image 1 Max val: {tf.reduce_max(img1_).numpy()}"
        )

        embeddings1_vit = vit_model(img1_).numpy()
        embeddings2_vit = vit_model(img2_).numpy()
        print('Vision Transformer embeddings:')
        print('- Shape:', embeddings1_vit.shape)
        print('- Dtype:', embeddings1_vit.dtype)
        print('- Mean:', tf.reduce_mean(embeddings1_vit))
        print('- Min:', tf.reduce_min(embeddings1_vit))
        print('- Max:', tf.reduce_max(embeddings1_vit))
        score_vit = compute_score(embeddings1_vit, embeddings2_vit)
        print('Vision Transformer score:')
        print('- Score:', score_vit)

        embeddings1_resnet = resnet50_model(img1_).numpy()
        embeddings2_resnet = resnet50_model(img2_).numpy()
        print('ResNet50 embeddings:')
        print('- Shape:', embeddings1_resnet.shape)
        print('- Dtype:', embeddings1_resnet.dtype)
        print('- Mean:', tf.reduce_mean(embeddings1_resnet))
        print('- Min:', tf.reduce_min(embeddings1_resnet))
        print('- Max:', tf.reduce_max(embeddings1_resnet))
        score_resnet = compute_score(embeddings1_resnet, embeddings2_resnet)
        print('ResNet50 score:')
        print('- Score:', score_resnet)

        embeddings1_vgg16 = vgg16_model(img1_).numpy()
        embeddings2_vgg16 = vgg16_model(img2_).numpy()
        print('VGG16 embeddings:')
        print('- Shape:', embeddings1_vgg16.shape)
        print('- Dtype:', embeddings1_vgg16.dtype)
        print('- Mean:', tf.reduce_mean(embeddings1_vgg16))
        print('- Min:', tf.reduce_min(embeddings1_vgg16))
        print('- Max:', tf.reduce_max(embeddings1_vgg16))
        score_vgg16 = compute_score(embeddings1_vgg16, embeddings2_vgg16)
        print('VGG16 score:')
        print('- Score:', score_vgg16)

        embeddings1_inception = inception_model(img1_).numpy()
        embeddings2_inception = inception_model(img2_).numpy()
        print('InceptionV3 embeddings:')
        print('- Shape:', embeddings1_inception.shape)
        print('- Dtype:', embeddings1_inception.dtype)
        print('- Mean:', tf.reduce_mean(embeddings1_inception))
        print('- Min:', tf.reduce_min(embeddings1_inception))
        print('- Max:', tf.reduce_max(embeddings1_inception))
        score_inception = compute_score(embeddings1_inception, embeddings2_inception)
        print('InceptionV3 score:')
        print('- Score:', score_inception)

        embeddings1_mobilenet = mobilenet_model(img1_).numpy()
        embeddings2_mobilenet = mobilenet_model(img2_).numpy()
        print('MobileNetV2 embeddings:')
        print('- Shape:', embeddings1_mobilenet.shape)
        print('- Dtype:', embeddings1_mobilenet.dtype)
        print('- Mean:', tf.reduce_mean(embeddings1_mobilenet))
        print('- Min:', tf.reduce_min(embeddings1_mobilenet))
        print('- Max:', tf.reduce_max(embeddings1_mobilenet))
        score_mobilenet = compute_score(embeddings1_mobilenet, embeddings2_mobilenet)
        print('MobileNetV2 score:')
        print('- Score:', score_mobilenet)

        embeddings1_efficientnet = efficientnetB0_model(img1_).numpy()
        embeddings2_efficientnet = efficientnetB0_model(img2_).numpy()
        print('EfficientNetB0 embeddings:')
        print('- Shape:', embeddings1_efficientnet.shape)
        print('- Dtype:', embeddings1_efficientnet.dtype)
        print('- Mean:', tf.reduce_mean(embeddings1_efficientnet))
        print('- Min:', tf.reduce_min(embeddings1_efficientnet))
        print('- Max:', tf.reduce_max(embeddings1_efficientnet))
        score_efficientnet = compute_score(embeddings1_efficientnet, embeddings2_efficientnet)
        print('EfficientNetB0 score:')
        print('- Score:', score_efficientnet)

        results[key] = {
            'vit': score_vit,
            'resnet': score_resnet,
            'vgg': score_vgg16,
            'inception': score_inception,
            'mobilenet': score_mobilenet,
            'efficientnet': score_efficientnet,
            'GT': val[-1]
        }

    with open('./saved_results/Tests/LFW/results.pickle', 'wb') as file:
        pickle.dump(results, file)

print(f"[INFO] Results:")
for key, val in results.items():
    print(f"[INFO] Pair {key} -> \tGround truth: {val['GT']}")
    print(f"[INFO] \t ViT: \t\t{round(val['vit'], 2)}")
    print(f"[INFO] \t ResNet: \t{round(val['resnet'], 2)}")
    print(f"[INFO] \t VGG: \t\t{round(val['vgg'], 2)}")
    print(f"[INFO] \t Inception: \t\t{round(val['inception'], 2)}")
    print(f"[INFO] \t MobileNet: \t\t{round(val['mobilenet'], 2)}")
    print(f"[INFO] \t Efficientnet: \t\t{round(val['efficientnet'], 2)}")

compute_roc(results, fig_name='ROC', positive_label=1)
